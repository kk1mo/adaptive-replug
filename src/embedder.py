import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm

def load_contriever():
    tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
    model = AutoModel.from_pretrained('facebook/contriever')
    model.eval()
    return model, tokenizer

def _mean_pooling(token_embeddings, mask):
    mask_expanded = mask.unsqueeze(-1).float()
    summed  = (token_embeddings * mask_expanded).sum(1)
    counted = mask_expanded.sum(1).clamp(min=1e-9)
    vecs    = summed / counted
    return F.normalize(vecs, p=2, dim=-1)

def embed_sentences(sentences, model, tokenizer, device, batch_size=64):
    all_vecs = []
    it = range(0, len(sentences), batch_size)
    it = tqdm(it, desc="Embedding sentences", unit="batch")
    for i in it:
        batch = sentences[i : i + batch_size]
        enc   = tokenizer(
            batch, padding=True, truncation=True,
            max_length=512, return_tensors="pt",
        ).to(device)
        out  = model(**enc)
        vecs = _mean_pooling(out.last_hidden_state, enc["attention_mask"])
        all_vecs.append(vecs.cpu().float().detach().numpy())
    return np.vstack(all_vecs)

# Example usage:
# >>> device = "cuda"
# >>> model, tokenizer = load_contriever()
# >>> model.to(device)
# >>> sentences = [
#   "Where was Marie Curie born?",
#   "Maria Sklodowska, later known as Marie Curie, was born on November 7, 1867.",
#   "Born in Paris on 15 May 1859, Pierre Curie was the son of Eugène Curie, a doctor of French Catholic origin from Alsace."
# ]
# >>> embeddings = embed_sentences(sentences, model, tokenizer, device=device)
# >>> print(embeddings)
# [[-0.01586545 -0.00137962 , ...], ...]