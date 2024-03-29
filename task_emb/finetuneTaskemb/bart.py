import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import T5ForConditionalGeneration, BartForConditionalGeneration
#from transformers.modeling_bart import shift_tokens_right
from transformers.models.bart.modeling_bart import shift_tokens_right

from utils import label_smoothed_nll_loss
#decoder_cached_states for older transformer version
class MyBart(BartForConditionalGeneration):
    def forward(self, input_ids, attention_mask=None, encoder_outputs=None,
            decoder_input_ids=None, decoder_attention_mask=None,
            output_hidden_states=True,return_dict=False,
            use_cache=False, is_training=False):

        if is_training:
            _decoder_input_ids = shift_tokens_right(decoder_input_ids, self.config.pad_token_id,decoder_start_token_id=self.config.decoder_start_token_id)
            #_decoder_input_ids = decoder_input_ids
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
        lm_logits = F.linear(outputs[0], self.model.shared.weight, bias=self.final_logits_bias)
        if is_training:
            lprobs = F.log_softmax(lm_logits, dim=-1)
            loss, _ = label_smoothed_nll_loss(lprobs, decoder_input_ids, epsilon=0.1, ignore_index=self.config.pad_token_id)
            return loss
        return lm_logits, outputs
