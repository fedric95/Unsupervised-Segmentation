# Unsupervised-Segmentation

Implementation of different Deep Learning Unsupervised Segmentation models in Pytorch (Lightning).

## ISB - Unsupervised Image Segmentation by Backpropagation

Asako Kanezaki. Unsupervised Image Segmentation by Backpropagation. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2018. (pdf)

Implementation based on: https://github.com/kanezaki/pytorch-unsupervised-segmentation

## DFC - Unsupervised Learning of Image Segmentation Based on Differentiable Feature Clustering

Wonjik Kim*, Asako Kanezaki*, and Masayuki Tanaka. Unsupervised Learning of Image Segmentation Based on Differentiable Feature Clustering. IEEE Transactions on Image Processing, accepted, 2020. (arXiv). *W. Kim and A. Kanezaki contributed equally to this work.

Implementation based on: https://github.com/kanezaki/pytorch-unsupervised-segmentation-tip

## WNet - A Deep Model for Fully Unsupervised Image Segmentation

Xia, Xide, and Brian Kulis. "W-net: A Deep Model for Fully Unsupervised Image Segmentation." arXiv preprint arXiv:1711.08506 (2017).

Implementation based on: https://aswali.github.io/WNet/

# Install

conda create -n hunan  python=3.7.10
conda activate hunan
conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.6 -c pytorch


pip install https://github.com/fedric95/Unsupervised-Segmentation.git

# Examples

In this repository, in the examples directory, there is an example for each method that has been implemented.

# TO-DO

ISB and DFC supports batch sizes grater than one but the computation is not efficient (it is not vectorized)