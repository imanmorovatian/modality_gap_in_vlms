# Mitigating the Modality Gap in Vision-Language Pre-Trained Models

## Overview
This repo contains the code related to a part of the work I conducted during my master's thesis. I proposed a fine-tunig method for the modality gap reduction in vision-language models.
Fine-tuning can be carried out using three different loss functions
* L<sub>clip</sub>
* L<sub>CUA</sub>
* L<sub>CUAXU</sub>

Based on the model, there are four different fine-tuning strategies
* LL
* LU
* UL
* UU

After fine-tuning, the effect of the gap reduction is examined through the performance of models on two downstream tasks

* cross-modality retrieval
* multimodal vector arithmetic.

Also, it is possible to measure the gap through two metrics

* Central Distance (CD): defines the modality gap as the difference between the center of image embeddings and text embeddings
* Central Moment Discrepancy (CMD): measures the distributional difference between two feature sets by comparing their higher-order moments

Furthermore, for the visualization, the code can provide the UMAP and box plots of image and text embeddings.

## Models

The following Visual-Language Models are supported

* ALBEF
* ALIGN
* CLIP (ViT-B/32)
* CLIP (ResNet50)
* CyCLIP
* FLAVA
* ImageBind
* VISTA

## Datasets

The following datasets are supported

* Flickr30k
* MSCOCO
* Conceptual Captions

## How to Run

### Environment
Create the environment using ```requirements.txt```

### Preprocess

1. create the following folders
<pre> <code>
   ├── data/

    │ ├── images/

      │ ├── flickr30k/

      │ ├── mscoco/

        │ ├── train2017/

        │ ├── val2017/

      | ├── visualGenom/

    │ ├── annotations/

      │ ├── flickr30k/

      │ ├── mscoco/

    │ ├── bpe/
</code> </pre>

2. Download the images of flickr30, MSCOCO (train and test 2017) and Visual Genom datasets. Put them in the corresponding folders.
3. Download the annotation files of flickr30 and MSCOCO (train and test 2017; captions.json). Put them in the corresponding folders. 
4. run ``` preprocess/flickr30k/split.py```
5. run ```preprocess/conceptualCaptions/sample.py``` and then  ```preprocess/conceptualCaptions/split.py```
6. run ```preprocess/simat/prepare_dataset.py```

### Fine-tune
run ```train.py``` with the required arguments. The required arguments are mentioned at the beginning of the file

### Cross-modality retrieval
run ```cross_retrieval.py``` with the required arguments. The required arguments are mentioned at the beginning of the file

### Multimodal vector arithmetic
run ```simat.py``` with the required arguments. The required arguments are mentioned at the beginning of the file

### Generate embedding for other uses
run ```compute_embeds.py``` with the required arguments. The required arguments are mentioned at the beginning of the file