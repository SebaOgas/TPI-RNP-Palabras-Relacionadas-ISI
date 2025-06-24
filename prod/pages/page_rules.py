import streamlit as st
import pandas as pd

st.title("ⓘ Reglas")

st.markdown("""
**ISINet Quizz** es un juego basado en la ISINet, que consiste en relacionar conceptos sobre temas de Ingeniería de Sistemas de Información. 

### 🎯 Ronda
Se presenta un concepto base y **el jugador deberá seleccionar los conceptos que considera que están más relacionados con el concepto base**. 
Mientras más cerca esté el concepto dentro del top 10 de las palabras más relacionadas, **más puntos obtendrá**.

### ❤️ Vidas
El juego es individual y **el jugador posee de 3 vidas**. Por cada intento fallido (el concepto seleccionado no pertenece al top 10) pierde una vida.
El jugador pierde el juego una vez perdida sus 3 vidas, por lo que deberá iniciar uno nuevo.
            
### 🏆 Puntuación
Cuando el jugador selecciona entre uno de los 10 conceptos más relacionados consigue sumar puntos. El cálculo
del puntaje procede según la posición en el top que ocupe el concepto ingresado, siendo la **puntuación máxima de 10 para el top 1** y la **mínima de 1 para el top 10**.
""")


if st.button("Jugar", icon=":material/sports_esports:", use_container_width=True):
    st.switch_page("pages/page_play.py")
