import streamlit as st
import os
import random
import pandas as pd
import time

st.title("Juego")
st.caption("¡Adivina las palabras relacionadas a temas de Ingeniería en Sistemas de Información!")

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
        vocabulary_path = "../data/vocabularies/vocab_0.txt"
        vocabulary = []
        
        with open(vocabulary_path, "rb") as cf:
            lines = cf.read().decode("utf-8").split("\n")[:-1]
            vocabulary = [ast.literal_eval(l) for l in lines]
        
        # Load model
        model_path = "../data/models/dataset-100-256-256.pt"
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
if 'vidas' not in st.session_state:
    st.session_state.vidas = 3

# Game interface
st.header("🎯 Encuentra las palabras relacionadas")

# Generar nuevo concepto si es necesario
if st.session_state.current_concept is None:
    concept_idx = random.randint(0, len(vocabulary) - 1)
    st.session_state.current_concept = concept_idx
    st.session_state.related_concepts = get_related_concepts(
                        concept_idx, 10, model.central_embedding, vocabulary, torch)

# Mostrar concepto actual
if st.session_state.current_concept is not None:
    current_concept = vocabulary[st.session_state.current_concept]
    
    # JUST FOR TESTING BLOCK
    os.write(1, "Top 10 conceptos más relacionados:\n".encode())
    # Show the related concepts
    for i, related_concept in enumerate(st.session_state.related_concepts, 1):
        concept_str = " + ".join(related_concept)
        os.write(1, f"{i}. {concept_str}\n".encode())
   # END OF JUST FOR TESTING BLOCK

    col1, col2 = st.columns(2, border=True)

    hearts = "❤️" * st.session_state.vidas + "🤍" * (3 - st.session_state.vidas)

    with col1:
        st.html(f"<div style='text-align:center; font-size:20px;'>🏆 Puntaje {st.session_state.score}</div>")

    with col2:
        st.html(f"<div style='text-align:center; font-size:20px;'>Vidas {hearts}</div>")

    st.header(f"🎮 Concepto -> {' + '.join(current_concept)}")

st.subheader("Tu turno")
    
# Play Again Btn Component
def play_again_btn(key):
    if st.button("Nuevo juego", type="secondary", key=key,use_container_width=True, icon=":material/sports_esports:"):
        st.session_state.current_concept = None
        st.session_state.related_concepts = []
        st.session_state.score = 0
        st.session_state.attempts = 0
        st.session_state.vidas = 3
        st.rerun()

# Loose Dialog Component
@st.dialog("🛑 Ya no te quedan más vidas")
def loose_dialog():
    play_again_btn("no-lifes")
    st.header("**Top 10 conceptos relacionados**")

    conceptos = []
    # Show the related concepts
    for related_concept in st.session_state.related_concepts:
        concept_str = " + ".join(related_concept)
        conceptos.append(concept_str)

    dt = pd.DataFrame({
        "puestos" : range(1, len(conceptos) + 1),
        "conceptos" : conceptos
    })

    st.dataframe(dt, hide_index=True, column_config= {
        "puestos": st.column_config.Column(
            "Puestos",
            width = "small",
        ),
        "conceptos": st.column_config.Column(
            "Conceptos",
            width = "large"
        )
    })

def get_guess_points(user_guess):
    # parsing user's input
    guess_words = [word.strip().lower() for word in user_guess.split(",")]
    guess_tuple = tuple(sorted(guess_words))

    for idx ,related_concept in enumerate(st.session_state.related_concepts):
        related_tuple = tuple(sorted([word.lower() for word in related_concept]))
        if guess_tuple == related_tuple:
            puntos = 10 - idx  # Top1 = 10 pts, Top10 = 1 pt
            return puntos  

    return 0 # player didn't guess          


player_lost = st.session_state.vidas == 0

if player_lost:
    loose_dialog()

col1, col2 = st.columns([12, 1], vertical_alignment="bottom")
user_guess = col1.text_input("Ingresa palabras separadas por comas que crees que están relacionadas:", 
                              placeholder="Ej: servidor, protocolo")

if col2.button("", disabled=player_lost, icon=":material/keyboard_return:"):
    
    
    points = get_guess_points(user_guess)
    st.session_state.score += points
    
    if points == 0:
        st.session_state.vidas = max(0, st.session_state.vidas - 1)
        st.error("No está en el top 10. ¡Intenta de nuevo!") #TODO -  no se estan mostrando estos msj por el rerun
        time.sleep(1.5)
        st.rerun()


    
    st.success(f"¡Correcto! Ganaste {points} puntos 🎉") #TODO -  no se estan mostrando estos msj por el rerun
    # Pasar a nuevo concepto
    st.session_state.current_concept = None
    time.sleep(1.5)
    st.rerun()
    
#TODO -  no se elimina el contenido del input luego de verificar la rta

# Reset game button
play_again_btn("always-shown")
