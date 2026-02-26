# Code adapted from https://huggingface.co/facebook/contriever

import torch
from transformers import AutoTokenizer, AutoModel

def load_contriever():
    tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
    model = AutoModel.from_pretrained('facebook/contriever')
    return model, tokenizer

def _mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings

def embed_sentences(sentences, model, tokenizer):
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    outputs = model(**inputs)

    embeddings = _mean_pooling(outputs[0], inputs['attention_mask'])
    return embeddings

# Example usage:
# >>> model, tokenizer = load_contriever()
# >>> sentences = [
#   "Where was Marie Curie born?",
#   "Maria Sklodowska, later known as Marie Curie, was born on November 7, 1867.",
#   "Born in Paris on 15 May 1859, Pierre Curie was the son of Eugène Curie, a doctor of French Catholic origin from Alsace."
# ]
# >>> embeddings = embed_sentences(sentences, model, tokenizer)
# >>> print(embeddings)
# tensor([[-0.0209, -0.0018, ...], ...])