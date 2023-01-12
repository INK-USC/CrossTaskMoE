#from transformers.modeling_bart import EncoderLayer, DecoderLayer, BartEncoder, BartDecoder, BartModel, BartForConditionalGeneration
#from transformers.modeling_bart import shift_tokens_right

from transformers.models.bart.modeling_bart import (
    BartEncoderLayer, BartDecoderLayer, BartEncoder, BartDecoder, 
    BartModel, BartForConditionalGeneration,
    shift_tokens_right
)
#/home/juanzha/transformers/src/transformers/models/bart/modeling_bart.py
#from transformers.configuration_bart import BartConfig
from transformers.models.bart.configuration_bart import BartConfig
from utils import label_smoothed_nll_loss

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import copy

class BartWithAdapterConfig(BartConfig):
    def __init__(
        self,
        activation_dropout=0.0,
        activation_function="gelu",
        vocab_size=50265,
        d_model=1024,
        encoder_ffn_dim=4096,
        encoder_layers=12,
        encoder_attention_heads=16,
        decoder_ffn_dim=4096,
        decoder_layers=12,
        decoder_attention_heads=16,
        encoder_layerdrop=0.0,
        decoder_layerdrop=0.0,
        attention_dropout=0.0,
        dropout=0.1,
        max_position_embeddings=1024,
        init_std=0.02,
        classifier_dropout=0.0,
        num_labels=3,
        is_encoder_decoder=True,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        normalize_before=False,
        add_final_layer_norm=False,
        scale_embedding=False,
        normalize_embedding=True,
        static_position_embeddings=False,
        add_bias_logits=False,
        adapter_dim=8,
        adapt_layer_norm=False,
        unfreeze_hyper_encoder=False,
        **common_kwargs
    ):

        if "hidden_size" in common_kwargs:
            raise ValueError("hidden size is called d_model")
        
        super().__init__(
            num_labels=num_labels,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            **common_kwargs,
        )
        self.vocab_size = vocab_size
        self.d_model = d_model  # encoder_embed_dim and decoder_embed_dim
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_layers = self.num_hidden_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_layerdrop = encoder_layerdrop
        self.decoder_layerdrop = decoder_layerdrop
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.init_std = init_std  # Normal(0, this parameter)
        self.activation_function = activation_function

        # Params introduced for Mbart
        self.scale_embedding = scale_embedding  # scale factor will be sqrt(d_model) if True
        self.normalize_embedding = normalize_embedding  # True for mbart, False otherwise
        self.normalize_before = normalize_before  # combo of fairseq's encoder_ and decoder_normalize_before
        self.add_final_layer_norm = add_final_layer_norm

        # Params introduced for Marian
        self.add_bias_logits = add_bias_logits
        self.static_position_embeddings = static_position_embeddings

        # 3 Types of Dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.dropout = dropout

        # Classifier stuff
        self.classif_dropout = classifier_dropout

        # Adapter
        self.adapter_dim = adapter_dim
        self.adapt_layer_norm = adapt_layer_norm

def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    #nn.init.xavier_uniform_(m.weight, gain=0.0000001)
    nn.init.ones_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m

class EncoderLayerWithAdapter(BartEncoderLayer):

    def __init__(self, config: BartConfig):
        super(EncoderLayerWithAdapter, self).__init__(config)
        #super().__init__()

        self.adapter_dim = config.adapter_dim
        #print("self.adapter_dim",self.adapter_dim)
        self.adapter_down0 = Linear(self.embed_dim, self.adapter_dim)
        self.adapter_up0 = Linear(self.adapter_dim, self.embed_dim)
        self.adapter_down1 = Linear(self.embed_dim, self.adapter_dim)
        self.adapter_up1 = Linear(self.adapter_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_head_mask: torch.Tensor,
        output_attentions: bool = False,
        output_fc=None,
        output_res=None,
    	):
        residual = hidden_states
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

		# START adapter 0
        residual_adapter = hidden_states
        hidden_states = self.adapter_down0(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.adapter_up0(hidden_states)
        hidden_states = residual_adapter + hidden_states
        # END adapter 0

        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

		# START adapter 1
        residual_adapter = hidden_states
        hidden_states = self.adapter_down1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.adapter_up1(hidden_states)
        hidden_states = residual_adapter + hidden_states
        # END adapter 1

        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)
        return outputs

class DecoderLayerWithAdapter(BartDecoderLayer):
    def __init__(self, config: BartConfig):
        super(DecoderLayerWithAdapter, self).__init__(config)

        self.adapter_dim = config.adapter_dim

        self.adapter_down0 = Linear(self.embed_dim, self.adapter_dim)
        self.adapter_up0 = Linear(self.adapter_dim, self.embed_dim)
        self.adapter_down1 = Linear(self.embed_dim, self.adapter_dim)
        self.adapter_up1 = Linear(self.adapter_dim, self.embed_dim)
        self.adapter_down2 = Linear(self.embed_dim, self.adapter_dim)
        self.adapter_up2 = Linear(self.adapter_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
        output_fc: Optional[bool] = False,
        output_res: Optional[bool] = False,
   		):
        residual = hidden_states

        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

		# START adapter 0
        residual_adapter = hidden_states
        hidden_states = self.adapter_down0(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.adapter_up0(hidden_states)
        hidden_states = residual_adapter + hidden_states
        # END adapter 0

        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            
            # START adapter 1
            residual_adapter = hidden_states
            hidden_states = self.adapter_down1(hidden_states)
            hidden_states = self.activation_fn(hidden_states)
            hidden_states = self.adapter_up1(hidden_states)
            hidden_states = residual_adapter + hidden_states
        	# END adapter 1

            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value
		
        # Fully Connected
        residual = hidden_states
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # START adapter 2
        residual_adapter = hidden_states
        hidden_states = self.adapter_down2(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.adapter_up2(hidden_states)
        hidden_states = residual_adapter + hidden_states
        # END adapter 2

        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)
        if use_cache:
            outputs += (present_key_value,)

        return outputs


class BartEncodeWithAdapter(BartEncoder):
    def __init__(self, config: BartConfig, embed_tokens):
        super(BartEncodeWithAdapter, self).__init__(config, embed_tokens)
        self.layers = nn.ModuleList(
            [EncoderLayerWithAdapter(config) for _ in range(config.encoder_layers)]
        )

class BartDecoderWithAdapter(BartDecoder):
    def __init__(self, config: BartConfig, embed_tokens: nn.Embedding):
        super(BartDecoderWithAdapter, self).__init__(config, embed_tokens)
        self.layers = nn.ModuleList(
            [DecoderLayerWithAdapter(config) for _ in range(config.decoder_layers)]
        )

class BartModelWithAdapter(BartModel):
    def __init__(self, config: BartConfig):
        super(BartModelWithAdapter, self).__init__(config)
        self.encoder = BartEncodeWithAdapter(config, self.shared)
        self.decoder = BartDecoderWithAdapter(config, self.shared)   

class BartForConditionalGenerationWithAdapter(BartForConditionalGeneration):
    def __init__(self, config: BartConfig):
        super().__init__(config)
        base_model = BartModelWithAdapter(config)
        self.model = base_model
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))

class MyBartWithAdapter(BartForConditionalGenerationWithAdapter):
    def forward(self, input_ids, attention_mask=None, encoder_outputs=None,
            decoder_input_ids=None, decoder_attention_mask=None, 
            output_hidden_states=True,return_dict=False,
            use_cache=False, is_training=False):

        if is_training:
            _decoder_input_ids = shift_tokens_right(decoder_input_ids, self.config.pad_token_id,self.config.decoder_start_token_id)
        else:
            _decoder_input_ids = decoder_input_ids

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=_decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=use_cache,
        )
        lm_logits_adapter = F.linear(outputs[0], self.model.shared.weight, bias=self.final_logits_bias)
        if is_training:
            lprobs = F.log_softmax(lm_logits_adapter, dim=-1)
            loss, _ = label_smoothed_nll_loss(lprobs, decoder_input_ids, epsilon=0.1, ignore_index=self.config.pad_token_id)
            return loss
        return (lm_logits, ) + outputs[1:]
    
    def encoders(self):
        return self.model.encoder.layers

    def decoders(self):
        return self.model.decoder.layers
