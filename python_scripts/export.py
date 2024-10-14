import onnxruntime as ort
import onnx
import copy

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
input_text = "What is your favorite"
inputs = tokenizer(input_text, return_tensors='pt')
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']
with torch.no_grad():
    outputs = model(input_ids, attention_mask=attention_mask, use_cache=True)
logits = outputs.logits
past_key_values = outputs.past_key_values
new_input_text = " sport?\n"
new_inputs = tokenizer(new_input_text, return_tensors='pt')
new_input_ids = new_inputs['input_ids']
extended_attention_mask = torch.cat(
    [attention_mask, torch.ones((attention_mask.size(0), new_input_ids.size(-1)), dtype=attention_mask.dtype)], dim=-1
)


class GPT2Wrapper(torch.nn.Module):
    def __init__(self, model):
        super(GPT2Wrapper, self).__init__()
        self.model = model
        self.num_layers = 12

    def forward(self, input_ids, attention_mask, past_key_values):

        # Reshape past_key_values from (num_layers * 2, batch_size, num_heads, seq_length, head_dim)
        if torch.all(past_key_values == 0):
            past_key_values = None
        else:
            past_key_values = tuple(
                (past_key_values[i], past_key_values[i + 1])
                for i in range(0, 2 * self.num_layers, 2)
            )

        outputs = self.model(input_ids, attention_mask=attention_mask,
                             past_key_values=past_key_values, use_cache=True)
        output_logits = outputs.logits
        past_key_values_flat = torch.cat(
            [torch.stack(pk, dim=0) for pk in outputs.past_key_values], dim=0)
        return output_logits, past_key_values_flat

    def freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False


wrapper = GPT2Wrapper(model)
wrapper.freeze_parameters()
past_key_values_flat = torch.cat(
    [torch.stack(pk, dim=0) for pk in past_key_values], dim=0)
print(past_key_values_flat.shape)
torch.onnx.export(
    wrapper,
    (new_input_ids, extended_attention_mask, past_key_values_flat),
    "gpt2_with_kv.onnx",
    input_names=["input_ids", "attention_mask", "past_key_values_input"],
    # Include past_key_values_output
    output_names=["logits", "past_key_values_output"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "new_sequence"},
        "attention_mask": {0: "batch_size", 1: "new_and_last_sequence"},
        "logits": {0: "batch_size"},  # Update dynamic axes for logits
        "past_key_values_input": {1: "batch_size", 3: "last_sequence"},
        # Include dynamic axes for past_key_values_output
        "past_key_values_output": {1: "batch_size", 3: "new_and_last_sequence"}
    },
    opset_version=14
)
