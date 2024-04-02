# TRANSFER LEARNING

Here we use pretrained model to classify images.

- First we use our pretrained autoencoder to extract features from MNIST10 images. Then we use these features to train a classifier. --> `train.py`
- Second we use a pretrained resnet50 from torchvision.models to classify images from CIFAR10 dataset. --> `train_imagenet.py`

## Refer

- [Transfer Learning](https://lightning.ai/docs/pytorch/stable/advanced/transfer_learning.html)