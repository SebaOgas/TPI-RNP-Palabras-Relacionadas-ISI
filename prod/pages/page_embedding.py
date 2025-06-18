import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.graph_objs import *
from sklearn.decomposition import PCA
import os
import ast

st.title("Embeddings")
st.caption("Visualiza nuestro embedding!")

def idx_to_label(idx):
    file = "../data/vocabulary.txt"
    with open(file, "rb") as cf:
        lines = cf.read().decode("utf-8").split("\n")[:-1]
        vocabulary = [ast.literal_eval(l) for l in lines]
    return vocabulary[idx]

def load_data():
    file = "../data/embeddings.txt"
    df = pd.read_table(file, sep=' ', header=None)
    data = df.values.tolist()

    # first index of each row contains the label
    labels = [idx_to_label(int(d[0])) for d in data]
    data = np.array([d[1:] for d in data])
    return data, labels

data, labels = load_data()

def plot_3D(data, labels):
    sizes = [5]*len(labels)
    colors = ['rgb(93, 164, 164)']*len(labels)

    fig = go.Figure(data=[go.Scatter3d(
                    x=data[:,0], y=data[:,1], z=data[:,2],
                    mode='markers',
                    text=labels,
                    marker=dict(
                        color=colors,
                        size=sizes
                    )
                )], layout=Layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'))
    return fig

def render_plot(fig):
        fig.update_layout(margin={"r":50,"t":100,"l":0,"b":0}, height=750, width=850)
        st.plotly_chart(fig)


pca = PCA(n_components=3)
data = pca.fit_transform(data)

fig = plot_3D(data, labels)
render_plot(fig)

