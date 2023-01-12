import torch
import torch.nn as nn
import torch.nn.functional as F

import itertools
import random

from typing import Optional, Tuple, Dict, Any

from transformers.models.bart.modeling_bart import (
    BartLearnedPositionalEmbedding, 
    BartEncoderLayer, BartDecoderLayer, BartEncoder, BartDecoder, 
    BartModel, BartForConditionalGeneration,
    shift_tokens_right, _expand_mask
)
from transformers.modeling_outputs import (
    BaseModelOutput, 
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqModelOutput,
    Seq2SeqLMOutput,
)
from transformers.file_utils import ModelOutput

from .router_v2 import RouterWrapper
from .routing_bart_config import RoutingBartConfig
from .utils import label_smoothed_nll_loss

class RoutingBartEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_blocks = config.router_block_num
        self.blocks = nn.ModuleList(
            [BartEncoderLayer(config) for _ in range(self.num_blocks)]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_head_mask: torch.Tensor,
        block_distribution: torch.Tensor = None,
        output_attentions: bool = False,
        use_sparse: bool = False,
    ):
        if use_sparse:
            selected_expert_idx = torch.argmax(block_distribution[0,:]).item()

            output = self.blocks[selected_expert_idx](
                hidden_states, attention_mask, layer_head_mask, output_attentions
            )

            return output

        else:
            outputs = torch.stack(
                [self.blocks[i](hidden_states, attention_mask, layer_head_mask, output_attentions)[0]
                    for i in range(self.num_blocks)],
                dim=-1
            )

            # BartEncoderLayer always output a tuple,
            # so there is "[0]" in the stack above,
            # and we also return a tuple.
            # The same applys to the decoders.

            # outputs -> bsz * seqlen * hidden_dim * experts
            # block distribution -> bsz * experts
            # print(outputs.shape)
            # print(block_distribution.shape)
            x = torch.einsum('blde,be->bld', outputs, block_distribution)

            return (x, )

    def get_weight_regularization_loss(self):
        combinations = list(itertools.combinations(range(self.num_blocks), 2))
        total_loss = 0.0
        for exp1_id, exp2_id in combinations:
            exp1, exp2 = self.blocks[exp1_id], self.blocks[exp2_id]
            all_losses = []
            for p1, p2 in zip(exp1.parameters(), exp2.parameters()):
                total_loss += torch.norm(p1-p2)
        total_loss = total_loss / len(combinations)
        return total_loss

class RoutingBartDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_blocks = config.router_block_num
        self.blocks = nn.ModuleList(
            [BartDecoderLayer(config) for _ in range(self.num_blocks)]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        encoder_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
        use_sparse: bool = False,
        block_distribution: torch.Tensor = None,
    ):

        # use_cache is only used in model.generate()
        # we only compute with the used expert
        if use_cache and use_sparse:

            selected_expert_idx = torch.argmax(block_distribution[0,:]).item()

            output = self.blocks[selected_expert_idx](
                hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask,
                layer_head_mask, encoder_layer_head_mask, past_key_value, output_attentions, use_cache
            )

            return output

        elif use_cache and (not use_sparse):
            if past_key_value is None:
                # should only be called for the first layer decoder
                past_key_value = [None] * self.num_blocks

            # get output from all experts in this layer
            expert_outputs = [self.blocks[i](
                hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask,
                layer_head_mask, encoder_layer_head_mask, past_key_value[i], output_attentions, use_cache
            )
                for i in range(self.num_blocks)]
            
            # get the hidden states ("[0]")
            outputs = torch.stack(
                [expert_output[0]
                    for expert_output in expert_outputs],
                dim=-1
            )

            # reorganize the cache so that they can be re-used later
            cache = tuple([expert_output[3 if output_attentions else 1] for expert_output in expert_outputs])

            x = torch.einsum('blde,be->bld', outputs, block_distribution)

            return (x, cache)
            
        else:
            outputs = torch.stack([self.blocks[i](
                hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask,
                layer_head_mask, encoder_layer_head_mask, past_key_value, output_attentions, use_cache
            )[0]
                for i in range(self.num_blocks)],
                dim=-1
            )

            x = torch.einsum('blde,be->bld', outputs, block_distribution)

            return (x, )

    def get_weight_regularization_loss(self):
        combinations = list(itertools.combinations(range(self.num_blocks), 2))
        total_loss = 0.0
        for exp1_id, exp2_id in combinations:
            exp1, exp2 = self.blocks[exp1_id], self.blocks[exp2_id]
            all_losses = []
            for p1, p2 in zip(exp1.parameters(), exp2.parameters()):
                total_loss += torch.norm(p1-p2)
        total_loss = total_loss / len(combinations)
        return total_loss

class RoutingBartEncoder(BartEncoder):      

    def __init__(self, config: RoutingBartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
        )

        self.layers = nn.ModuleList()
        for idx in range(config.encoder_layers):
            if idx in self.config.encoder_vanilla_layers:
                self.layers.append(BartEncoderLayer(config))
            else:
                self.layers.append(RoutingBartEncoderLayer(config))

        self.layernorm_embedding = nn.LayerNorm(embed_dim)

        self.init_weights()       

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        block_distribution=None,
        use_sparse=None,
    ):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        embed_pos = self.embed_positions(input_shape)

        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        # print(hidden_states.shape)

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            if head_mask.size()[0] != (len(self.layers)):
                raise ValueError(
                    f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
                )

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

            # use vanilla transformer layer (no expert or routing)
            if idx in self.config.encoder_vanilla_layers:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    output_attentions=output_attentions,
                )
                
            # use routing and experts
            else:
                block_distribution_for_current_layer = block_distribution[idx]

                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    output_attentions=output_attentions,
                    block_distribution=block_distribution_for_current_layer, # feed the block distribution info to the encoder layer
                    use_sparse=use_sparse,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)
        
        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


class RoutingBartDecoder(BartDecoder):

    def __init__(self, config: RoutingBartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
        )

        self.layers = nn.ModuleList()
        for idx in range(config.encoder_layers):
            if idx in self.config.encoder_vanilla_layers:
                self.layers.append(BartDecoderLayer(config))
            else:
                self.layers.append(RoutingBartDecoderLayer(config))

        self.layernorm_embedding = nn.LayerNorm(config.d_model)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        head_mask=None,
        encoder_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        use_sparse=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        decoder_block_distribution=None,
    ):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])

        # embed positions
        positions = self.embed_positions(input_shape, past_key_values_length)

        hidden_states = inputs_embeds + positions
        hidden_states = self.layernorm_embedding(hidden_states)

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if idx in self.config.decoder_vanilla_layers:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    encoder_layer_head_mask=(encoder_head_mask[idx] if encoder_head_mask is not None else None),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                )
            
            else:
                block_distribution_for_current_layer = decoder_block_distribution[idx]

                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    encoder_layer_head_mask=(encoder_head_mask[idx] if encoder_head_mask is not None else None),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    use_sparse=use_sparse,
                    block_distribution=block_distribution_for_current_layer,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)        

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )

class RoutingBartModel(BartModel):
    def __init__(self, config: RoutingBartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = RoutingBartEncoder(config, self.shared)
        self.decoder = RoutingBartDecoder(config, self.shared)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        use_sparse=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        block_distribution=None,
        decoder_block_distribution=None,
    ):

        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                block_distribution=block_distribution,
                use_sparse=use_sparse,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            encoder_head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            use_sparse=use_sparse,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            decoder_block_distribution=decoder_block_distribution,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

class Selection(nn.Module):
    def __init__(self, mode, tau=5.0):
        super().__init__()

        self.mode = mode
        self.tau = tau

    def forward(self, x, temperature=None):
        if self.mode == "softmax":
            if temperature is not None:
                x = F.softmax(x / temperature, dim=-1)
            else:
                x = F.softmax(x / self.tau, dim=-1)
        elif self.mode == "gumbel_softmax":
            x = F.gumbel_softmax(x, tau=self.tau, hard=False)
        else: # gumbel_softmax_st
            x = F.gumbel_softmax(x, tau=self.tau, hard=True)
        return x
    
    def set_gumbel_temperature(self, tau):
        self.tau = tau

class MyRoutingBart(BartForConditionalGeneration):
    def __init__(self, config: RoutingBartConfig):
        super().__init__(config)
        self.router = RouterWrapper(config)
        self.model = RoutingBartModel(config)
        self.selection = Selection(config.router_mode)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))

        self.init_weights()
    
    def forward(
        self, 
        input_ids, 
        attention_mask=None, 
        encoder_outputs=None,
        decoder_input_ids=None, 
        decoder_attention_mask=None, 
        use_cache=False, 
        use_sparse=False,
        is_training=False,
        task_embed=None,
        block_distribution=None,
        decoder_block_distribution=None,
        **model_kwargs,
    ):

        if task_embed is not None:
            # get routes
            enc_route, dec_route = self.router(task_embed)
            # (gumbel) softmax
            block_distribution = self.selection(enc_route)
            decoder_block_distribution = self.selection(dec_route)
            if is_training and self.config.exploration_epsilon > 0.0:
                block_distribution, decoder_block_distribution = self._posthoc_process(block_distribution, decoder_block_distribution)
        else:
            assert block_distribution is not None and decoder_block_distribution is not None

        if is_training:
            _decoder_input_ids = shift_tokens_right(decoder_input_ids, self.config.pad_token_id, self.config.decoder_start_token_id)
        else:
            _decoder_input_ids = decoder_input_ids

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=_decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            use_cache=use_cache,
            use_sparse=use_sparse,
            block_distribution=block_distribution,
            decoder_block_distribution=decoder_block_distribution,
            **model_kwargs,
        )

        lm_logits = F.linear(outputs[0], self.model.shared.weight, bias=self.final_logits_bias)
        if is_training:
            lprobs = F.log_softmax(lm_logits, dim=-1)
            loss, _ = label_smoothed_nll_loss(lprobs, decoder_input_ids, epsilon=0.1, ignore_index=self.config.pad_token_id)
            return loss

        return Seq2SeqLMOutput(
            loss=None,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def _posthoc_process(self, enc_routes, dec_routes):
        n_expert = self.config.router_block_num

        mask = torch.rand(enc_routes.shape[0], enc_routes.shape[1])
        selected = mask < self.config.exploration_epsilon
        n_selected = torch.sum(torch.sum(selected)).item()

        if n_selected != 0:
            # to_fill = torch.ones(n_selected, n_expert) / n_expert
            to_fill = torch.nn.functional.one_hot(torch.rand(n_selected, n_expert).argmax(dim=1),n_expert).float()
            enc_routes[selected] = to_fill.to(device=enc_routes.device)

        mask = torch.rand(dec_routes.shape[0], dec_routes.shape[1])
        selected = mask < self.config.exploration_epsilon
        n_selected = torch.sum(torch.sum(selected)).item()

        if n_selected != 0:
            to_fill = torch.ones(n_selected, n_expert) / n_expert
            # to_fill = torch.nn.functional.one_hot(torch.rand(n_selected, n_expert).argmax(dim=1),n_expert).float()
            dec_routes[selected] = to_fill.to(device=enc_routes.device)

        return enc_routes, dec_routes

    def set_router_mode(self, mode):
        assert mode in ["softmax", "gumbel_softmax", "gumbel_softmax_st"]    
        self.selection.mode = mode


    def get_routes(self, task_embed, separate=False, override=None, temperature=None):
        if len(task_embed.shape) == 1:
            task_embed = task_embed.unsqueeze(0)
        enc_, dec_ = self.router(task_embed)

        if override:
            enc_route_mask, dec_route_mask = override
            enc_ = enc_ - 1e10 * enc_route_mask
            dec_ = dec_ - 1e10 * dec_route_mask
        
        if separate:
            return self.selection(enc_.squeeze(1), temperature), self.selection(dec_.squeeze(1), temperature)
        else:
            return self.selection(torch.cat([enc_, dec_], dim=0).squeeze(1), temperature)

    def set_gumbel_temperature(self, tau):
        self.selection.set_gumbel_temperature(tau)

    def get_weight_regularization_loss(self):
        total_loss = 0.0
        for idx in range(self.config.encoder_layers):
            if idx not in self.config.encoder_vanilla_layers:
                total_loss += self.model.encoder.layers[idx].get_weight_regularization_loss()
        for idx in range(self.config.decoder_layers):
            if idx not in self.config.decoder_vanilla_layers:
                total_loss += self.model.decoder.layers[idx].get_weight_regularization_loss()
        return total_loss      

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        use_cache=False,
        use_sparse=False,
        encoder_outputs=None,
        task_embed=None,
        block_distribution=None,
        decoder_block_distribution=None,
        **kwargs
    ):

        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "task_embed": task_embed,
            "block_distribution": block_distribution,
            "decoder_block_distribution": decoder_block_distribution,
            "use_cache": use_cache, 
            "use_sparse": use_sparse,
        }

    @staticmethod
    # this is to deal with sequence expansion in beam search
    # input_ids gets replicated by num_beams times during decoding
    # so the routes should be replicated in a similar way
    def _expand_inputs_for_generation(
        input_ids: torch.LongTensor,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        attention_mask: torch.LongTensor = None,
        encoder_outputs: ModelOutput = None,
        **model_kwargs,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:

        expanded_return_idx = (
            torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1).to(input_ids.device)
        )

        input_ids, model_kwargs = RoutingBartModel._expand_inputs_for_generation(
            input_ids=input_ids,
            expand_size=expand_size,
            is_encoder_decoder=is_encoder_decoder,
            attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            **model_kwargs
        )

        # here decoder_block_distribution is replicated for num_beams times along dim 1
        # decoder_block_distribution has the shape of layer x bsz x blocks
        decoder_block_distribution = model_kwargs["decoder_block_distribution"]
        model_kwargs["decoder_block_distribution"] = decoder_block_distribution.index_select(1, expanded_return_idx)

        return input_ids, model_kwargs

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            if isinstance(layer_past[0], tuple): # use_cache=True; use_sparse=False
                reordered_past += ([
                    tuple(past_state.index_select(0, beam_idx) for past_state in expert_past[:2]) + expert_past[2:] for expert_past in layer_past
                ],)
            else: # use_cache=True; use_sparse=True; cache is in the same format as original bart
                reordered_past += (
                    tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
                )
        return reordered_past
