import streamlit as st
st.caption("Top 10 palabras relacionadas a temas de Ingeniería en Sistemas de Información")
if st.button("Jugar", icon=":material/sports_esports:", use_container_width=True):
    st.switch_page("pages/page_play.py")

# La verdad que no lo veo a streamlit muy sólido como para manejar estados fácilmente
if st.button("Reglas del juego", icon=":material/info:", use_container_width=True):
    st.switch_page("pages/page_rules.py")
