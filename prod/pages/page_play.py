import streamlit as st
import os
import random
import pandas as pd
import time
from itertools import cycle

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
        vocabulary_path = "data/vocabularies/vocab_0.txt"
        vocabulary = []
        
        with open(vocabulary_path, "rb") as cf:
            lines = cf.read().decode("utf-8").split("\n")[:-1]
            vocabulary = [ast.literal_eval(l) for l in lines]
        
        # Load model
        model_path = "prod/modelo.pt"
        model = SkipGram(vocabulary, 256)
        model.load_state_dict(torch.load(model_path, weights_only=True, map_location='cpu'))
        model.eval()
        
        return model, vocabulary, torch
        
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

def get_related_concepts_with_distractors(concept_ix, num_related_concepts, num_distractors, embed, vocabulary, torch_module):
    """Get k most related concepts to a given concept"""
    W = embed.weight.data
    x = W[torch_module.tensor(concept_ix)]

    cos = torch_module.mv(W, x) / torch_module.sqrt(torch_module.sum(W * W, dim=1) *
                                      torch_module.sum(x * x) + 1e-9)
    topk = torch_module.topk(cos, k=num_related_concepts+1+num_distractors+20)[1].cpu().numpy().astype('int32')

    related = []
    distractors = []
    distractors_pool = []
    added = 0

    for i in topk[1:]:
        if i < len(vocabulary):  # Safety check
            if added < num_related_concepts:
                related.append(vocabulary[i])
                added += 1 
            else:
                distractors_pool.append(vocabulary[i])
   
    distractors = random.sample(distractors_pool, min(num_distractors, len(distractors_pool))) 
 
    return related, distractors

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
if 'distractors' not in st.session_state:
    st.session_state.distractors = []
if 'score' not in st.session_state:
    st.session_state.score = 0
if 'attempts' not in st.session_state:
    st.session_state.attempts = 0
if 'vidas' not in st.session_state:
    st.session_state.vidas = 3
if 'selected_buttons' not in st.session_state:
    st.session_state.selected_buttons = {}
if 'all_options' not in st.session_state:
    st.session_state.all_options = []
if 'feedback_msg' not in st.session_state:
    st.session_state.feedback_msg = None
if 'aciertos' not in st.session_state:
    st.session_state.aciertos = 0
if 'clicked_concepts' not in st.session_state:
    st.session_state.clicked_concepts = set()

# Game interface
st.header("🎯 Encuentra las palabras relacionadas")

# Generar nuevo concepto si es necesario
if st.session_state.current_concept is  None:
    concept_idx = random.randint(0, len(vocabulary) - 1)
    st.session_state.current_concept = concept_idx
    st.session_state.related_concepts, st.session_state.distractors= get_related_concepts_with_distractors(
                                                                     concept_idx, 10, 6, model.central_embedding, vocabulary, torch)
    st.session_state.all_options = st.session_state.related_concepts + st.session_state.distractors
    random.shuffle(st.session_state.all_options)
# Mostrar concepto actual
if st.session_state.current_concept is not None:
    current_concept = vocabulary[st.session_state.current_concept]

    
    col1, col2 = st.columns(2, border=True)

    hearts = "❤️" * st.session_state.vidas + "🤍" * (3 - st.session_state.vidas)

    with col1:
        st.html(f"<div style='text-align:center; font-size:20px;'>🏆 Puntaje {st.session_state.score}</div>")

    with col2:
        st.html(f"<div style='text-align:center; font-size:20px;'>Vidas {hearts}</div>")

    st.header(f"🎮 Concepto -> {' + '.join(current_concept)}")

    
# Play Again Btn Component
def play_again_btn(key):
    if st.button("Nuevo juego", type="secondary", key=key,use_container_width=True, icon=":material/sports_esports:"):
        st.session_state.current_concept = None
        st.session_state.related_concepts = []
        st.session_state.distractors = []
        st.session_state.score = 0
        st.session_state.attempts = 0
        st.session_state.vidas = 3
        st.session_state.selected_buttons = {}
        st.session_state.all_options = []
        st.session_state.feedback_msg = None
        st.session_state.aciertos = 0
        st.session_state.clicked_concepts = set()
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
@st.dialog("🎉 ¡Adivinaste los 10 conceptos correctamente!")
def win_dialog():
    st.success("¡Excelente! Acertaste los 10 conceptos relacionados sin perder todas las vidas.")
    play_again_btn("win-game")

if (st.session_state.aciertos == len(st.session_state.related_concepts)) and st.session_state.vidas > 0:
    win_dialog()

player_lost = st.session_state.vidas == 0

if player_lost:
    loose_dialog()
# Reset game button
play_again_btn("always-shown")

st.subheader("Tu turno")

def handle_button_click(button_id, concept):
    st.session_state.selected_buttons[button_id] = True
    concept_key = tuple(concept)    
    if concept_key in st.session_state.clicked_concepts:
        st.session_state.feedback_msg = ("error", "Ya seleccionaste este concepto.")
        return
    st.session_state.clicked_concepts.add(concept_key)
    if concept in st.session_state.related_concepts:
        idx = st.session_state.related_concepts.index(concept)
        points = 10 - idx  # Top 1 = 10 pts, Top 10 = 1 pt
        st.session_state.score += points
        st.session_state.aciertos += 1   #Sumamos un acierto
        st.session_state.feedback_msg = ("success", f"¡Correcto! Ganaste {points} puntos 🎉")
    else:
        st.session_state.vidas = max(0, st.session_state.vidas - 1)
        st.session_state.feedback_msg = ("error", "No está en el top 10. ¡Intenta de nuevo!")
        
    # pausa breve antes de la próxima ronda
    time.sleep(1)
    
    
# Columnas para los botones
cols = st.columns(4)
for i, concept in enumerate(st.session_state.all_options):
            button_id = f"btn_{i}"
            already_clicked = st.session_state.selected_buttons.get(button_id, False)
            concept_already_clicked = tuple(concept) in st.session_state.clicked_concepts
            label = " + ".join(concept)

            col = cols[i % 4]
            with col:
                st.button(
                    label,
                    key=button_id,
                    type="secondary",
                    use_container_width=True,
                    disabled=already_clicked or concept_already_clicked or st.session_state.vidas == 0 or (st.session_state.aciertos == len(st.session_state.related_concepts)),
                    on_click=handle_button_click,
                    args=(button_id, concept),
                )

if st.session_state.feedback_msg is not None:
        tipo, mensaje = st.session_state.feedback_msg
        if tipo == "success":
            st.success(mensaje)
        elif tipo == "error":
            st.error(mensaje)
