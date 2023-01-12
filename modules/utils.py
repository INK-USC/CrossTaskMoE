import torch
import torch.nn as nn

def label_smoothed_nll_loss(lprobs, target, epsilon=0.1, ignore_index=-100):
    """From fairseq"""
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)

    nll_loss = nll_loss.sum()
    smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss

def initialize_weights(config, model_new, model_old, eps=1e-8):
    """model 1 should be the routing version of model 2.
    we will be use weights in model 2 to initialize model 1.
    when copying we apply small noise (characterized by eps)."""

    # embeddings
    model_new.model.shared.load_state_dict(model_old.model.shared.state_dict())
    model_new.model.encoder.embed_positions.load_state_dict(model_old.model.encoder.embed_positions.state_dict())
    model_new.model.decoder.embed_positions.load_state_dict(model_old.model.decoder.embed_positions.state_dict())

    # layer norms
    model_new.model.encoder.layernorm_embedding.load_state_dict(model_old.model.encoder.layernorm_embedding.state_dict())
    model_new.model.decoder.layernorm_embedding.load_state_dict(model_old.model.decoder.layernorm_embedding.state_dict())

    # encoders
    for layer in range(config.encoder_layers):
        old_encoder_layer = model_old.model.encoder.layers[layer]

        if layer in config.encoder_vanilla_layers:
            new_encoder_layer = model_new.model.encoder.layers[layer]
            new_encoder_layer.load_state_dict(old_encoder_layer.state_dict())

        else:
            # iterate blocks
            for block_idx in range(config.router_block_num):

                # copy weights from the old model
                new_encoder_layer = model_new.model.encoder.layers[layer].blocks[block_idx]
                new_encoder_layer.load_state_dict(old_encoder_layer.state_dict())

                # add noise (reference: https://discuss.pytorch.org/t/is-there-any-way-to-add-noise-to-trained-weights/29829)
                with torch.no_grad():
                    for param in new_encoder_layer.parameters():
                        param.add_(torch.randn(param.size()) * eps)

    # decoders
    for layer in range(config.decoder_layers):
        old_decoder_layer = model_old.model.decoder.layers[layer]

        if layer in config.decoder_vanilla_layers:
            new_decoder_layer = model_new.model.decoder.layers[layer]
            new_decoder_layer.load_state_dict(old_decoder_layer.state_dict())
        else:
            # iterate blocks
            for block_idx in range(config.router_block_num):

                # copy weights from the old model
                new_decoder_layer = model_new.model.decoder.layers[layer].blocks[block_idx]
                new_decoder_layer.load_state_dict(old_decoder_layer.state_dict())

                # add noise (reference: https://discuss.pytorch.org/t/is-there-any-way-to-add-noise-to-trained-weights/29829)
                with torch.no_grad():
                    for param in new_decoder_layer.parameters():
                        param.add_(torch.randn(param.size()) * eps)

def squeeze_weights(config, model_new, model_old, enc_routes, dec_routes):
    # new is a vanilla model
    # old is a routing model
    # enc_routes in shape [layer, n_experts]

    # embeddings
    model_new.model.shared.load_state_dict(model_old.model.shared.state_dict())
    model_new.model.encoder.embed_positions.load_state_dict(model_old.model.encoder.embed_positions.state_dict())
    model_new.model.decoder.embed_positions.load_state_dict(model_old.model.decoder.embed_positions.state_dict())

    # layer norms
    model_new.model.encoder.layernorm_embedding.load_state_dict(model_old.model.encoder.layernorm_embedding.state_dict())
    model_new.model.decoder.layernorm_embedding.load_state_dict(model_old.model.decoder.layernorm_embedding.state_dict())

    # encoders
    for layer in range(config.encoder_layers):
        new_encoder_layer = model_new.model.encoder.layers[layer]

        selected_expert_idx = torch.argmax(enc_routes[layer,:]).item()
        old_encoder_layer = model_old.model.encoder.layers[layer].blocks[selected_expert_idx]

        new_encoder_layer.load_state_dict(old_encoder_layer.state_dict())

    # decoders
    for layer in range(config.decoder_layers):
        new_decoder_layer = model_new.model.decoder.layers[layer]

        selected_expert_idx = torch.argmax(dec_routes[layer,:]).item()
        old_decoder_layer = model_old.model.decoder.layers[layer].blocks[selected_expert_idx]

        new_decoder_layer.load_state_dict(old_decoder_layer.state_dict())
