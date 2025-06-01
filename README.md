# Palabras Relacionadas de Ingeniería en Sistemas de Información

Este proyecto fue desarrollado para el Trabajo Práctico Integrador de la materia Redes Neuronales Profundas, Ingeniería en Sistemas de Información de la Universidad Tecnológica Nacional Facultad Regional Mendoza.

Consiste en un juego en que, basándose en la terminología propia de la carrera, se debe adivinar qué palabras una red neuronal considera que se encuentran relacionadas a otras.

## Estructura del Proyecto

* data/: contiene el dataset, tanto sin procesar como ya procesado, junto al script utilizado para procesarlo.
* dev/: contiene los scripts y Jupyter notebooks utilizados durante el desarrollo.
* prod/: código preparado para el entorno de producción.

## Uso

1. Clonar el repositorio

```
    git clone https://github.com/SebaOgas/TPI-RNP-Palabras-Relacionadas-ISI.git
```

2. Instalar dependencias

```
    pip install -r prod/requirements.txt
```

3. Ejecutar aplicación

```
    streamlit run prod\app.py
```