# pix2pix for Maps to Satellite Images Translation

Course: Neural Networks @ FIIT STU\
Authors: Viktor Modroczký & Michaela Hanková

The goal of this project is to implement the pix2pix model and train it on the [pix2pix Maps](http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/maps.tar.gz) dataset to translate maps to satellite images. We used Kaggle Notebooks with Python 3.7.12 to train and test the model utilizing the free GPU provided by Kaggle. The requirements for the project are listed in the [`requirements.txt`](./requirements.txt) file.

## Dataset

The dataset used for this project is the paired [pix2pix Maps](http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/maps.tar.gz) dataset containing 1096 training images and 1098 validation images. The dataset contains images of maps and their corresponding satellite images extracted from Google Maps. The dataset is also available on Kaggle [here](https://www.kaggle.com/datasets/alincijov/pix2pix-maps).

### Dataset Preprocessing

The functions used to preprocess the dataset are in the [`src`](./src) directory of this repository in the [`pixutils.py`](./src/pixutils.py) script. The functions are `load_image`, `extract_patches`, `random_jitter`, and `rescale_images`.

The map to satellite pairs are in one jpeg file, side-by-side. The size of the images is 600x1200. For this reason, the images need to be split into two separate images. We defined a custom `tf.data.Dataset` pipeline to load the images and split them into two separate 600x600 images. This step is done right after loading the image file in a function called `load_image` which is passed to the `map` function of the dataset pipeline.

The 600x600 are relatively large images, so we incorporated a step to the pipeline to extract 256x256 patches from the images. This step is done in the `extract_patches` function which is also passed to the `map` function of the dataset pipeline. This outputs a dataset of 1096 and 1098 batches of four 256x256 patches for the training and validation sets respectively. For this reason we use the `unbatch` function to get a dataset of 4384 and 4392 patches for the training and validation sets respectively.

Next we introduce random jitter to the training images by resizing them to 286x286 and then randomly cropping them back to 256x256. After that, we horizontally flip the images with a 50% chance. This is done in the `random_jitter` passed to the `map` function of the dataset pipeline.

Finally, we normalize the images to the range [-1, 1] by dividing them by 127.5 and subtracting 1. This is done in the `rescale_images` function passed to the `map` function of the dataset pipeline.

The training dataset is shuffled and batched with a batch size of 1. The validation dataset is also batched with a batch size of 1.

The random jitter step was inspired by the [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004) paper by Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, Alexei A. Efros (2016). The random jitter step is also used in the [TensorFlow pix2pix tutorial](https://www.tensorflow.org/tutorials/generative/pix2pix). The aforementioned paper also recommends a batch size of 1 for the pix2pix model and for GANs in general.

## Model

The model used for this project is the pix2pix model introduced in the [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004) paper. The model consists of a generator and a discriminator. The generator is a U-Net based architecture that is fully convolutional and doesn't use any pooling layers like the original U-Net architecture ([U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597), Olaf Ronneberger, Philipp Fischer, and Thomas Brox, 2015). The discriminator is a PatchGAN model introduced by the authors of the pix2pix paper.

The models and their training loop are defined in the [`src`](./src) directory of this repository in the [`pix2pix.py`](./src/pix2pix.py) script. The generator and discriminator models are defined in the `UNet` and `PatchGAN` functions respectively. The training loop is defined in the `fit` function. The loss functions are defined in the `g_loss` and `d_loss` functions for the generator and discriminator respectively.
