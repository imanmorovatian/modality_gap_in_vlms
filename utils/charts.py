import os
import numpy as np
import pandas as pd
import umap
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm


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

def draw_boxplot(model_names: list[str], dataset: str):
    points = []
    types = []
    models = []

    for model in tqdm(model_names, total=len(model_names)):

        text_embedding = torch.load(f'results/embeddings/{model}/{dataset}/text.pt',
                                    map_location=torch.device('cpu')).numpy()
        
        image_embedding = torch.load(f'results/embeddings/{model}/{dataset}/image.pt',
                                     map_location=torch.device('cpu')).numpy()

        all_sim_img_txt, all_dissim_img_txt = similarities(image_embedding @ text_embedding.T)

        all_sim_img_txt = all_sim_img_txt.tolist()
        all_dissim_img_txt = all_dissim_img_txt.tolist()

        points += all_sim_img_txt
        types += ['similarity'] * len(all_sim_img_txt)
        models += [model] * len(all_sim_img_txt)

        points += all_dissim_img_txt
        types += ['dissimilarity'] * len(all_dissim_img_txt)
        models += [model] * len(all_dissim_img_txt)


    data = {
        'point': points,
        'type': types,
        'model': models
    }

    sns.set_theme(rc={'figure.figsize':(18,15)})
    ax = sns.boxplot(data, x='model', y='point', hue='type')
    ax.set(
        xlabel=None,
        ylabel=None,
        )
    ax.tick_params(axis='x', labelrotation=45)

    result_dir = 'results/plots/boxplots'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    img_name = ','.join(model_names)
    img_name += f'_{dataset}.jpg'

    plt.savefig(f'{result_dir}/{img_name}')
    print(f'Saved {img_name} successfully in {result_dir}')

def draw_umap(model_names: list[str], dataset: str, n_row: int, n_col: int,
              n_neighbors: int = 10, min_dist: int = 0.2, spread: int = 1.2):
    
    fig, axes = plt.subplots(n_row, n_col, figsize=(15, 20))
    axes = axes.flatten()

    for idx, model in enumerate(tqdm(model_names, total=len(model_names))):

        image_embeddings = torch.load(f'results/embeddings/{model}/{dataset}/image.pt',
                                    map_location=torch.device('cpu')).numpy()
        
        text_embeddings = torch.load(f'results/embeddings/{model}/{dataset}/text.pt',
                                    map_location=torch.device('cpu')).numpy()

        
        embeddings = np.concatenate([image_embeddings, text_embeddings], axis=0)
        labels = np.array([0] * len(image_embeddings) + [1] * len(text_embeddings))

        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            n_components=2,
            metric='cosine',
            min_dist=min_dist,
            spread=spread
            )

        umap_results = reducer.fit_transform(embeddings)

        ax = axes[idx]
        for label, color, marker in zip([0, 1], ['blue', 'red'], ['o', 'x']):
            ax.scatter(
                umap_results[labels == label, 0],
                umap_results[labels == label, 1],
                label='Image' if label == 0 else 'Text',
                c=color,
                marker=marker,
                alpha=0.7
            )

        ax.set_title(model)
        ax.legend()
        
        if idx % n_col == 0:
            ax.set_ylabel('UMAP Dimension 2')

        if idx >= len(model_names) - ((n_row-1) * n_col):
            ax.set_xlabel('UMAP Dimension 1')


    # Hide any unused subplots
    for i in range(len(model_names), len(axes)):
        fig.delaxes(axes[i])

    # automatically adjust the spacing between subplots in a figure
    plt.tight_layout()

    result_dir = 'results/plots/umap'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    img_name = ','.join(model_names)
    img_name += f'_{dataset}.jpg'

    plt.savefig(f'{result_dir}/{img_name}')
    print(f'Saved {img_name} successfully in {result_dir}')