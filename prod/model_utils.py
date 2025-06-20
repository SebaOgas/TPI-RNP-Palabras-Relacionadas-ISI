import torch
from torch import nn
import ast
import streamlit as st

class SkipGram(nn.Module):
    def __init__(self, vocabulary, embed_size):
        super().__init__()
        self.central_embedding = nn.Embedding(num_embeddings=len(vocabulary)+1,
                                embedding_dim=embed_size, padding_idx=len(vocabulary))
        self.context_embedding = nn.Embedding(num_embeddings=len(vocabulary)+1,
                                embedding_dim=embed_size, padding_idx=len(vocabulary))

    def forward(self, center, contexts_and_negatives):
        v = self.central_embedding(center)
        u = self.context_embedding(contexts_and_negatives)
        pred = torch.bmm(v, u.permute(0, 2, 1))
        return pred

@st.cache_resource
def load_model_and_vocabulary():
    """Load the trained model and vocabulary with caching"""
    # Load vocabulary
    vocabulary_path = "../data/vocabulary.txt"
    vocabulary = []
    
    with open(vocabulary_path, "rb") as cf:
        lines = cf.read().decode("utf-8").split("\n")[:-1]
        vocabulary = [ast.literal_eval(l) for l in lines]
    
    # Load model
    model_path = "../data/models/100-1-256-256.pt"
    model = SkipGram(vocabulary, 256)
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location='cpu'))
    model.eval()
    
    return model, vocabulary

def get_related_concepts(concept_ix, k, embed, vocabulary):
    """Get k most related concepts to a given concept"""
    W = embed.weight.data
    x = W[torch.tensor(concept_ix)]

    cos = torch.mv(W, x) / torch.sqrt(torch.sum(W * W, dim=1) *
                                      torch.sum(x * x) + 1e-9)
    topk = torch.topk(cos, k=k+1)[1].cpu().numpy().astype('int32')

    related = []
    for i in topk[1:]:
        if i < len(vocabulary):  # Safety check
            related.append(vocabulary[i])
    return related

def search_concepts_containing_word(word, vocabulary, max_results=10):
    """Search for concepts that contain a specific word"""
    word = word.lower().strip()
    results = []
    
    for i, concept in enumerate(vocabulary):
        concept_words = [w.lower() for w in concept]
        if word in concept_words:
            results.append((i, concept))
            if len(results) >= max_results:
                break
    
    return results 