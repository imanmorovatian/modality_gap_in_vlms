# Modality-Invariance Evaluation for Visual-Language Models (VLMs)

## Overview

This project focuses on evaluating the modality-invariance of representations derived from Visual-Language Models (VLMs) on two widely used datasets: MSCOCO and Flickr30k. The goal is to assess the ability of these models to generate similar representations for different modalities (text and image). The evaluated models include CLIP (ViT-B/32, ResNet50), ALIGN, ALBEF, Florence, Uni-Perceiver, and ImageBind.

## Models

The following Visual-Language Models (VLMs) are evaluated in this project:

1. **CLIP (ViT-B/32)**
2. **CLIP (ResNet50)**
3. **ALIGN**
4. **ALBEF**
5. **Florence**
6. **Uni-Perceiver**
7. **ImageBind**

## Datasets

The evaluation is performed on two datasets:

1. **MSCOCO**
2. **Flickr30k**

## Evaluation Metrics

### 1. Similarity Distribution

Boxplots are generated to visualize the distribution of similarities among positive and negative pairs for different modalities, including:

- Text-Text
- Image-Image
- Image-Text

### 2. t-SNE Visualization

t-SNE (t-Distributed Stochastic Neighbor Embedding) plots are created to visualize the relationships and clustering of features in both images and text.

## How to Run

Follow these steps to reproduce the evaluation:

1. Clone the repository:

```bash
git clone https://github.com/your-username/your-repository.git
cd your-repository