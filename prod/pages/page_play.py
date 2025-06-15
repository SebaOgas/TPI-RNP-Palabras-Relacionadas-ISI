import streamlit as st
import os
import random

st.title("ISINet Quizz - Juego")
st.caption("¡Adivina las palabras relacionadas a temas de Ingeniería en Sistemas de Información!")

# Add a back button to return to main page
if st.button("Volver al inicio", icon=":material/arrow_back:", use_container_width=True):
    st.switch_page("streamlit_app.py")

@st.cache_resource
def load_model_components():
    """Load model components with lazy imports to avoid watcher issues"""
    try:
        # Import torch only inside this cached function
        import torch
        from torch import nn
        import ast
        
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
        
        return model, vocabulary, torch
        
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

def get_related_concepts(concept_ix, k, embed, vocabulary, torch_module):
    """Get k most related concepts to a given concept"""
    W = embed.weight.data
    x = W[torch_module.tensor(concept_ix)]

    cos = torch_module.mv(W, x) / torch_module.sqrt(torch_module.sum(W * W, dim=1) *
                                      torch_module.sum(x * x) + 1e-9)
    topk = torch_module.topk(cos, k=k+1)[1].cpu().numpy().astype('int32')

    related = []
    for i in topk[1:]:
        if i < len(vocabulary):  # Safety check
            related.append(vocabulary[i])
    return related

# Load model and vocabulary
model, vocabulary, torch = load_model_components()

if model is None or vocabulary is None or torch is None:
    st.error("No se pudo cargar el modelo. Verifica que los archivos estén en la ubicación correcta.")
    st.stop()

st.success("Modelo cargado exitosamente!")

# Session State data initialization
if 'current_concept' not in st.session_state:
    st.session_state.current_concept = None
if 'related_concepts' not in st.session_state:
    st.session_state.related_concepts = []
if 'score' not in st.session_state:
    st.session_state.score = 0
if 'attempts' not in st.session_state:
    st.session_state.attempts = 0

# Game interface
st.header("🎯 Encuentra las palabras relacionadas")

concept_idx = random.randint(0, len(vocabulary) - 1)
st.session_state.current_concept = concept_idx
st.session_state.related_concepts = get_related_concepts(
                    concept_idx, 10, model.central_embedding, vocabulary, torch
                )

if st.session_state.current_concept is not None:
    current_concept = vocabulary[st.session_state.current_concept]
    st.header(f"🎮 Concepto actual: {' + '.join(current_concept)}")
    
    """
    JUST FOR TESTING BLOCK
    """
    st.write("**Top 10 conceptos más relacionados:**")
    
    # Show the related concepts
    for i, related_concept in enumerate(st.session_state.related_concepts, 1):
        concept_str = " + ".join(related_concept)
        st.write(f"{i}. {concept_str}")

    """
    END OF JUST FOR TESTING BLOCK
    """
    
    st.subheader("Tu turno")
    user_guess = st.text_input("Ingresa palabras separadas por comas que crees que están relacionadas:", 
                              placeholder="Ej: servidor, protocolo")
    

    if user_guess and st.button("Verificar respuesta"):
        # parsing user's input
        guess_words = [word.strip().lower() for word in user_guess.split(",")]
        guess_tuple = tuple(sorted(guess_words))
        
        found = False
        for related_concept in st.session_state.related_concepts:
            related_tuple = tuple(sorted([word.lower() for word in related_concept]))
            if guess_tuple == related_tuple:
                found = True
                break
        
        st.session_state.attempts += 1
        
        if found:
            st.session_state.score += 1
            st.success("¡Correcto! 🎉")
        else:
            st.error("No está en el top 10. ¡Intenta de nuevo!")

# Reset game button
if st.button("Nuevo juego", type="secondary"):
    st.session_state.current_concept = None
    st.session_state.related_concepts = []
    st.session_state.score = 0
    st.session_state.attempts = 0
    st.rerun()
