# Recognize Handwritten Letters

## Description

The goal of this project is to recognize handwritten letters.

<p align="middle">
  <img src="assets/img/dataset-first-letters.png" />
</p>

We are working on a [MNIST-like dataset for letters](https://www.kaggle.com/datasets/ashishguptajiit/handwritten-az/data).

## Gallery

### Embedding of the letters with UMAP

![UMAP embedding of the letters](assets/img/umap-embedding-plot.png)

Here we reduced the dimension of the dataset from 784 dimensions to 2, with the UMAP algorithm.

### Learning curves & correlation matrix with a CNN network
<p>
  <img src="assets/img/cnn-training-curve.png" width="49%" />
  <img src="assets/img/cnn-confusion-matrix.png" width="49%" /> 
</p>

Our CNN achieved more than 98% of accuracy on the test set.
