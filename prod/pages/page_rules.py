import streamlit as st
import pandas as pd

st.title("ⓘ Reglas")

st.markdown("""
**ISINet Quizz** es un juego basado en la ISINet, que consiste en relacionar conceptos sobre temas de Ingeniería de Sistemas de Información. 

### Ronda
Se presenta un concepto a fin y **el jugador deberá ingresar conceptos relacionados**. Para pasar a la siguiente ronda, el concepto
relacionado deberá estar en el **top 10 de conceptos más relacionados al propuesto**.

### Vidas
El juego es individual y **el jugador posee de 3 vidas**. Por cada intento fallido (el concepto ingresado no pertenece al top 10) pierde una vida.
El jugador pierde el juego una vez perdida sus 3 vidas, por lo que deberá iniciar uno nuevo.
            
### Puntuación
Cuando el jugador acierta entre uno de los 10 conceptos más relacionados, pasa a la siguiente ronda y consigue sumar puntos. El cálculo
del puntaje procede según la posición en el top que ocupe el concepto ingresado, siendo la **puntuación máxima de 10 para el top 1** y la **mínima 1 para el top 10**.
""")


if st.button("Jugar", icon=":material/sports_esports:", use_container_width=True):
    st.switch_page("pages/page_play.py")
