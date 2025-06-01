from pypdf import PdfReader
import os
from os import listdir
from os.path import isfile, join


path = "./data/raw"

plain_path = "./data/plain.txt"

if os.path.exists(plain_path):
    os.remove(plain_path)

raw = [f for f in listdir(path) if isfile(join(path, f))]

metrics = {}
for f in raw:
    print("\033[94mConvirtiendo archivo: " + f + "\033[0m")

    sf = f.split("-")
    anio = sf[0].strip()
    materia = sf[1].strip()
    if (not anio in metrics):
        metrics[anio] = {}

    if (not materia in metrics[anio]):
        metrics[anio][materia] = 0

    metrics[anio][materia] += os.path.getsize("./data/raw/" + f)

    reader = PdfReader("./data/raw/" + f)
    for p in reader.pages:
        with open(plain_path, "ab") as t:
            t.write(p.extract_text(
                extraction_mode="plain", 
                layout_mode_space_vertically=False).encode("utf-8"))
    
print("\033[92m")
for a, materias in metrics.items():
    print("Año " + a + ":")
    for m, tamano in materias.items():
        print("\tMateria: " + m + " - " + str(round(tamano/1000000,2)) +"MB")
print("\033[0m")