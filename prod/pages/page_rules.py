import streamlit as st

st.title("ISINet Quizz - Reglas")

if st.button("Volver al inicio", icon=":material/arrow_back:", use_container_width=True):
    st.switch_page("streamlit_app.py")

st.markdown("""
**ISINet Quizz** es un juego que consiste en relacionar conceptos de temas de Ingeniería de Sistemas. 
            Se presenta un concepto y el jugador tiene que adivinar un concepto relacionado que al menos
            esté entre los **top 10 de conceptos más relacionados al propuesto**.

Se juega de forma individual y el jugador consta de n vidas, que son intentos para adivinar el concepto entre 
            el top 10.

## ¿Cómo jugar?
Se presenta un concepto al jugador, el jugador deberá ingresar el concepto
que crea ser el más relacionado a ese. 
            
Una vez ingresado se dan dos situaciones:
            
            
1. El jugador acierta entre una de los 10 conceptos más relacionados.
2. El jugador no acierta.
            

En la primera situación el jugador pasa a la siguiente ronda y suma un puntaje. Dicho puntaje se calcula en función de que tan relacionado fue el concepto
            ingresado con el concepto de la ronda.
            
            

Cálculo de los puntajes:

* Top 1 = 10 puntos,

* Top 2 = 9 puntos,

 ...

* Top 10 = 1 punto.
 
            
En la segunda situación, el jugador pierde una vida. Cada jugador tiene 3 vidas. Si pierde sus 3 vidas entonces termina el juego, sino, pasa a la
siguiente ronda.

            """)