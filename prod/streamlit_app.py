import streamlit as st

ascii_art = """
██╗███████╗██╗     ██████╗ ██╗   ██╗██╗███████╗███████╗     ⠀⠀⠀⠀⠀⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
██║██╔════╝██║    ██╔═══██╗██║   ██║██║╚══███╔╝╚══███╔╝     ⠀⠀⢀⣴⣿⣿⣷⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀
██║███████╗██║    ██║   ██║██║   ██║██║  ███╔╝   ███╔╝      ⠀⠀⢸⣿⣿⠿⢿⣿⣿⣷⡀⠀⠀⠀⠀⠀⠀
██║╚════██║██║    ██║▄▄ ██║██║   ██║██║ ███╔╝   ███╔╝       ⠀⠀⣿⣿⡇⠀⠈⠙⣿⣿⡄⠀⠀⠀⠀⠀⠀
██║███████║██║    ╚██████╔╝╚██████╔╝██║███████╗███████╗     ⠀⠀⢿⣿⣷⣶⣶⣾⣿⣿⠇⠀⠀⠀⠀⠀⠀
╚═╝╚══════╝╚═╝     ╚══▀▀═╝  ╚═════╝ ╚═╝╚══════╝╚══════╝     ⠀⠀⠈⠛⠿⣿⣿⠿⠛⠁                                                                    
"""

st.code(ascii_art, language="text")
pages = {
    "General": [
        st.Page("pages/page_home.py", title="Incio", icon=":material/home:")
    ],
    "Juego": [
        st.Page("pages/page_play.py", title="Jugar", icon=":material/sports_esports:"),
        st.Page("pages/page_rules.py", title="Reglas", icon=":material/info:"),
    ], 
    "Extras": [

    ]
   
}

pg = st.navigation(pages)
pg.run()



