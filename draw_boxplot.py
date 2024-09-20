import os
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from tqdm import tqdm
from utils.chart_utils import similarities


model_names = None
dataset = None

result_dir = 'results/charts'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

points = []
types = []
models = []

for model in tqdm(model_names, total=len(model_names)):
    text_embedding = torch.load(f'results/embeddings/{dataset}/{model}/text.pt').cpu().numpy()
    image_embedding = torch.load(f'results/embeddings/{dataset}/{model}/image.pt').cpu().numpy()

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
    title='Flickr Dataset',
    xlabel=None,
    ylabel=None,
    )
ax.tick_params(axis='x', labelrotation=45)
plt.savefig(result_dir+f'/{dataset}.jpg')
print('Saved the image successfully')