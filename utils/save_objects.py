import os
import plotly.graph_objects as go
from typing import List


def save_objects(gobjs: List, outfolder: str, name: str):
    fig = go.Figure()
    for gobj in gobjs:
        fig.add_trace(gobj)

    fig.update_layout(
        boxmode='group',
        font=dict(size=58),
        yaxis_title = "Cosine similarity",
        autosize    = False,
        width       = 1000,
        height      = 1000,
    )

    fig.update_layout(yaxis_range=[-0.2,1])
    fig.write_image(os.path.join(outfolder, f'{name}.png'), scale=2)
    fig.write_html(os.path.join(outfolder, f'{name}.html'))