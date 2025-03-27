# copied from @philschmid
# https://github.com/philschmid/deep-learning-pytorch-huggingface/blob/main/training/utils/llama_patch.py

# flash decoder work copied from
# https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/


import torch
import transformers
from peft.tuners.lora import LoraLayer

try:
    pass
except Exception:
    raise ModuleNotFoundError(
        "Please install FlashAttention first, e.g., with pip install flash-attn --no-build-isolation, Learn more at https://github.com/Dao-AILab/flash-attention#installation-and-features"
    )

try:
    pass
except Exception:
    raise ModuleNotFoundError("Please install einops first, e.g., with pip install einops")


# Disable the transformation of the attention mask in LlamaModel as the flash attention
# requires the attention mask to be the same as the key_padding_mask
def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
    # [bsz, seq_len]
    return attention_mask


def flash_attn_forward_without_dropout(
    self,
    hidden_states,
    attention_mask=None,
    position_ids=None,
    past_key_value=None,
    output_attentions=False,
    use_cache=False,
    **kwargs,
):
    original_fwd = transformers.models.llama.modeling_llama.LlamaModel.LlamaFlashAttention2.forward
    original_training_status = self.training
    self.training = False
    result = original_fwd(
        self,
        hidden_states,
        attention_mask,
        position_ids,
        past_key_value,
        output_attentions,
        use_cache,
        **kwargs,
    )
    self.training = original_training_status
    return result


def replace_attn_with_flash_attn(disable_dropout: bool = False):
    cuda_major, cuda_minor = torch.cuda.get_device_capability()
    transformers.models.llama.modeling_llama.LlamaModel._prepare_decoder_attention_mask = (
        _prepare_decoder_attention_mask
    )
    if disable_dropout:
        transformers.models.llama.modeling_llama.LlamaModel.LlamaFlashAttention2.forward = (
            flash_attn_forward_without_dropout
        )


# Adapted from https://github.com/tmm1/axolotl/blob/2eda9e02a9d15a7a3f92b41f257d9844d72fc220/src/axolotl/utils/models.py#L338
def upcast_layer_for_flash_attention(model, torch_dtype):
    # LlamaRMSNorm layers are in fp32 after kbit_training, so we need to
    # convert them back to fp16/bf16 for flash-attn compatibility.
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            module.to(torch_dtype)
        if "norm" in name:
            module.to(torch_dtype)
        if "lm_head" in name or "embed_tokens" in name:
            if hasattr(module, "weight"):
                module.to(torch_dtype)

    return model


# Disable the transformation of the attention mask in LlamaModel as the flash attention
# requires the attention mask to be the same as the key_padding_mask
def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
    return attention_mask
