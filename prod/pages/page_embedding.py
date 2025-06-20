import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.graph_objs import *
from sklearn.manifold import TSNE 
import os
import ast

st.title("Embeddings")
st.caption("Visualiza nuestro embedding!")

def idx_to_label(idx):
    file = "../data/vocabularies/vocab_0.txt"
    with open(file, "rb") as cf:
        lines = cf.read().decode("utf-8").split("\n")
        vocabulary = [ast.literal_eval(l) for l in lines]
    return vocabulary[idx]

def load_data(dim):
    file = f"../data/embeddings/embeddings-{dim}.csv"
    data = np.loadtxt(file, delimiter=",").tolist()
    
    # first index of each row contains the label
    labels = [idx_to_label(int(d[0])) for d in data]
    data = np.array([d[1:] for d in data])
    return data, labels

def plot_3D(data, labels, search_idx = None):
    tsne = TSNE(n_components=3)
    data = tsne.fit_transform(data)

    sizes = [5]*len(labels)
    colors = ['rgb(93, 164, 214)']*len(labels)

    if search_idx:
        sizes[search_idx] = 15
        colors[search_idx] = 'rgb(243, 14, 114)'

    fig = go.Figure(data=[go.Scatter3d(
                    x=data[:,0], y=data[:,1], z=data[:,2],
                    mode='markers',
                    text=labels,
                    marker=dict(
                        size=sizes,
                        color=colors
                    )
                )])
    return fig

def plot_2D(data, labels, search_idx = None):
    tsne = TSNE(n_components=2)
    data = tsne.fit_transform(data)
    
    sizes = [5]*len(labels)
    colors = ['rgb(93, 164, 214)']*len(labels)

    if search_idx:
        sizes[search_idx] = 15
        colors[search_idx] = 'rgb(243, 14, 114)'

    fig = go.Figure(data=[go.Scatter(
                x=data[:,0], y=data[:,1],
                mode='markers',
                text=labels,
                marker=dict(
                    size=sizes,
                    color=colors,
                ),
            )])
    
    return fig

def render_plot(fig):
        fig.update_layout(height=750, width=850)
        st.plotly_chart(fig)

def plot_for_D(data, labels, dim, search_idx = None):
    if dim == "2d":
        fig = plot_2D(data, labels, search_idx)
    else:
        fig = plot_3D(data, labels, search_idx)
    render_plot(fig)


st.sidebar.header(":material/settings: Configuración Embedding")

dim = st.sidebar.radio(
    "Dimensiones",
    ["2d", "3d"]
)

search_for = st.sidebar.text_input("Buscar concepto", placeholder="word1, word2")

if st.sidebar.button("Visualize", icon=":material/graph_3:"):
    data, labels = load_data(dim)
    if search_for:
        try:
            concept = tuple([word.strip() for word in search_for.split(',')])
            search_idx = labels.index(concept)
            plot_for_D(data, labels, dim, search_idx)
        except ValueError:
            st.sidebar.error("Couldnt find " +  search_for)

    plot_for_D(data, labels, dim)