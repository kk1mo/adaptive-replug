import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def load_GPT2(model_name, device):
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    gpt2_model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    gpt2_model.eval()
    gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
    # gpt2_maxlength = gpt2_model.config.n_positions
    return gpt2_model, gpt2_tokenizer


@torch.no_grad()
def score_gpt2(model, tokenizer, prefix, continuation, device):
    prefix_ids = tokenizer.encode(prefix)
    continuation_ids = tokenizer.encode(continuation)

    input_ids = torch.tensor([prefix_ids + continuation_ids]).to(device)
    with torch.no_grad():
        logits = model(input_ids).logits

    log_probs = F.log_softmax(logits[0], dim=-1)
    cont_start = len(prefix_ids)
    cont_tensor = torch.tensor(continuation_ids, dtype=torch.long).to(device)
    lp = log_probs[cont_start - 1 : cont_start - 1 + len(continuation_ids)]
    score = lp.gather(1, cont_tensor.unsqueeze(1)).sum().item()
    return score

# Example usage:
# >>> m, t = load_GPT2("gpt2", "cuda")
# >>> print(score_gpt2(m, t, "The capital of France is ", "Paris.", "cuda"))
# -16.95