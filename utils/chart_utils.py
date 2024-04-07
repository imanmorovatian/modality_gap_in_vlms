import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def tsne_2dplot(X: np.ndarray, y: np.ndarray, categories: np.ndarray=None):
    """
    Generate a 2D t-SNE plot for visualizing embeddings.

    Args:
    - X (np.ndarray): Input data matrix.
    - y (np.ndarray): Labels corresponding to the input data.

    Returns:
    - fig: Plotly figure object.
    """
    scaler     = StandardScaler()
    X = scaler.fit_transform(X, y=y)

    X_embedded        = TSNE(n_components=2, learning_rate='auto',
                    init='random').fit_transform(X)

    # print(X_embedded)

    df           = pd.DataFrame()
    df["Modality"]      = ['Text' if i==1.0 else 'Image' for i in y]
    df["x"] = X_embedded[:,0]
    df["y"] = X_embedded[:,1]

    if categories:
        df['ImageNet Class'] = categories
        fig = px.scatter(df, x="x", y="y",
                    symbol='Modality',  # Use 'Category' as the symbol attribute
                    color='ImageNet Class',   # Use 'Category' for coloring points
                    size_max=30,
                    # markers=dict(
                    #     Image='circle',  # Use 'circle' marker for category A
                    #     Text='square'   # Use 'square' marker for category B
                    # ),
        )
    else:
        fig = px.scatter(df, x="x", y="y",
                        symbol='Modality',  # Use 'Category' as the symbol attribute
                        color='Modality',   # Use 'Category' for coloring points
                        size_max=30,
                        # markers=dict(
                        #     Image='circle',  # Use 'circle' marker for category A
                        #     Text='square'   # Use 'square' marker for category B
                        # ),
        )

    # fig.update_layout(
    #     autosize    = False,
    #     width       = 1000,
    #     height      = 1000,
    #     plot_bgcolor  ='rgba(0,0,0,0)',
    #     font        = dict(
    #     family      = "Calibri",
    #     size        = 55,)
    # )
    
    return fig, X_embedded

def off_diag(matrix):
    """
    Extract off-diagonal elements from a square matrix.

    Args:
    - matrix: Input square matrix.

    Returns:
    - np.ndarray: Off-diagonal elements.
    """
     
    off_diagonal_elements = []
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[0]):
            if i>j: # lower matrix triangle
                off_diagonal_elements.append(matrix[i,j])
    return np.array(off_diagonal_elements)


def similarities(sim_matrix):
    """
    Extract diagonal and off-diagonal elements from a square similarity matrix.

    Args:
    - sim_matrix: Input square similarity matrix.

    Returns:
    - np.ndarray: Diagonal elements.
    - np.ndarray: Off-diagonal elements.
    """

    return np.diagonal(sim_matrix), off_diag(sim_matrix)

def boxplot(vector, pair_type, name:str, color = 'indianred'):
    """
    Create a box plot for visualizing a vector based on pair types.

    Args:
    - vector: The data vector to be visualized.
    - pair_type: The type of pairs corresponding to the data vector (e.g., 'txt-txt', 'img-img', 'txt-img').
    - name (str): A label for the box plot (e.g., 'pos', 'neg').
    - color (str): The marker color for the box plot.

    Returns:
    - go.Box: Plotly Box object representing the box plot.
    """
    
    return go.Box(y=vector,
                  x=pair_type,
                  name=name,
                  marker_color=color)