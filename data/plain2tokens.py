import spacy
from spacy.symbols import ORTH
from spacy.symbols import NORM
import os
import re

# how can I do this code better?
def is_clean_token(token): 
    return not (
        token.is_punct or 
        token.is_space or 
        token.is_stop or 
        len(token.text) == 1)

rules_path = "./data/rules.txt"
tokens_path = "./data/tokens.txt"
plain_path = "./data/plain.txt"

if os.path.exists(tokens_path):
    os.remove(tokens_path)

# Descargar el modelo con: python -m spacy download es_core_news_sm
esp = spacy.load("es_core_news_sm")

# Cargar las reglas especiales: secuencias de texto que no deben ser separadas
with open(rules_path, "rb") as rf:
    rules_txt = rf.read().decode("utf-8")
    rules = [r.strip() for r in re.split(r'(?:\r\n)+', rules_txt)]
    for r in rules:
        esp.tokenizer.add_special_case(r, [{ORTH: r}])

# Leer el archivo plano, generar los tokens y escribirlos al archivo de tokens (un token por línea)
with open(plain_path, "rb") as pf:
    txt = pf.read().decode("utf-8")
    tokens = esp.tokenizer(txt)
    with open(tokens_path, "wb") as tf:
        for token in tokens:
            if (is_clean_token(token)):
                tf.write((token.text + "\n").encode("utf-8"))
            
