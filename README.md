# pix2pix for Maps to Satellite Images Translation

Course: Neural Networks @ FIIT STU\
Authors: Viktor Modroczký & Michaela Hanková

The goal of this project is to implement the pix2pix model and train it on the [pix2pix Maps](http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/maps.tar.gz) dataset to translate maps to satellite images. We used Kaggle Notebooks with Python 3.7.12 to train and test the model utilizing the free GPU provided by Kaggle - Nvidia Tesla P100. The requirements for the project are listed in the [`requirements.txt`](./requirements.txt) file.

## Dataset

The dataset used for this project is the paired [pix2pix Maps](http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/maps.tar.gz) dataset containing 1096 training images and 1098 validation images. The dataset contains images of maps and their corresponding satellite images extracted from Google Maps. It is also available on Kaggle [here](https://www.kaggle.com/datasets/alincijov/pix2pix-maps). The dataset is split into two directories - `train` and `val`.

### Dataset Preprocessing

The functions used to preprocess the dataset are in the [`src`](./src) directory of this repository in the [`pixutils.py`](./src/pixutils.py) script. The functions are `load_image`, `extract_patches`, `random_jitter`, `rescale_images`, and `resize_images`.

The map to satellite pairs are in one jpeg file, side-by-side. The size of the images is 600x1200. For this reason, the images need to be split into two separate images. We defined a custom [`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset) pipeline to load the images and split them into two separate 600x600 images. This step is done right after loading the image file in a function called `load_image` which is passed to the `map` function of the dataset pipeline.

The 600x600 are relatively large images, so we incorporated a step to the pipeline to extract 256x256 patches from the images. This step is done in the `extract_patches` function which is also passed to the `map` function of the dataset pipeline. This outputs a dataset of 1096 and 1098 batches of four 256x256 patches for the training and validation sets respectively. For this reason we use the `unbatch` method to get a dataset of 4384 and 4392 patches for the training and validation sets respectively. This way we are not limited to a batch size of 4 which was output by the `extract_patches` function.

If we don't want to extract patches, we can use the `resize_images` function instead that resizes the images to 256x256. This function is passed to the `map` function of the dataset pipeline instead of the `extract_patches` function.

Next we introduce random jitter to the training images by resizing them to 286x286 and then randomly cropping them back to 256x256. After that, we horizontally flip the training images with a 50% chance. This is done in the `random_jitter` passed to the `map` function of the dataset pipeline.

After that we normalize the images to the range [-1, 1] by dividing them by 127.5 and subtracting 1. This is done in the `rescale_images` function passed to the `map` function of the dataset pipeline.

Finally, the training dataset is shuffled and batched with a set batch size. The validation dataset is also batched with the same batch size but not shuffled.

We used the following pipelines for training and validation data in our experiments:

**Training data pipeline**: `load_image` -> `extract_patches` -> `unbatch` -> `random_jitter` -> `rescale_images` -> `shuffle` -> `batch`

**Validation data pipeline**: `load_image` -> `extract_patches` -> `unbatch` -> `rescale_images` -> `batch`

**Training data pipeline with no patch extraction**: `load_image` -> `resize_images` -> `random_jitter` -> `rescale_images` -> `shuffle` -> `batch`

**Validation data pipeline with no patch extraction**: `load_image` -> `resize_images` -> `rescale_images` -> `batch`

The random jitter step was inspired by the [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004) paper by Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, Alexei A. Efros (2016). Random jitter is also used in the [TensorFlow pix2pix tutorial](https://www.tensorflow.org/tutorials/generative/pix2pix). The aforementioned paper also recommends a batch size of 1 for the pix2pix model and for GANs in general. However, we also tried different batch sizes.

Examples of map and satellite extracted patch pairs from the preprocessed training dataset (images were rescaled to [0, 1] for visualization purposes):

![Example 1](./figures/train_images_1.png)

![Example 2](./figures/train_images_2.png)

## Model

The model used for this project is the pix2pix model introduced in the [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004) paper. The model consists of a generator and a discriminator. The generator is a U-Net based architecture that is fully convolutional and doesn't use any pooling layers unlike the original U-Net architecture ([U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597), Olaf Ronneberger, Philipp Fischer, and Thomas Brox, 2015). Instead, it uses strided convolutions for downsampling and strided transposed convolutions for upsampling. Both downsampling and upsampling are done by a factor of 2, therefore strides are set to 2. Upsampling blocks output ReLU activations while downsampling blocks output LeakyReLU activations with the slope coefficient set to 0.2. Kernel size is set to 4 for all convolutional layers. The last layer of the generator outputs tanh activations in a range of [-1, 1] to match the range of the input images. The activation map has a depth of 3 and spatial resolution to match the input images.

The generator only works on 256x256 images since it is fully convolutional and has skip connections that use concatenation to combine outputs of each block in the contracting path with corresponding inputs in the expansive path. To make the generator work on smaller or larger images, the generator architecture needs to be modified to have lesser or more blocks in the contracting and expansive paths since larger images need to be downsampled more and smaller images need to be downsampled less to reach 1x1 spatial resolution.

The discriminator is a PatchGAN model, introduced by the authors of the pix2pix paper, that classifies NxN patches of the input image as real or fake. Each activation in the output maps to an NxN receptive field in the input image. The authors found that the best results were achieved by using 70x70 patches so we used that size for our discriminator first, then we tried other sizes too. The other patch sizes can be 1, 16, and 286. In case of 1x1 patches, the discriminator classifies each pixel as real or fake, and in case of 286x286 patches, the discriminator classifies the whole image as real or fake.

The PatchGAN model implementation uses the same downsampling blocks as the generator. The last layer uses the same kernel size of 4 but strides are set to 1 as this layer only reduces depth and not spatial resolution. This layer outputs a single value for each patch which is the probability that the patch is real.

We also use batch normalization and dropout in some blocks following the pix2pix paper. The dropout rate is 0.5. As per the paper, we also use Gaussian weight initialization with a mean of 0 and standard deviation of 0.02. The optimizer used for training both the generator and discriminator is Adam with a learning rate of 0.0002 and beta values of 0.5 and 0.999 for the first and second moments respectively.

The models and their training loop are defined in the [`src`](./src) directory of this repository in the [`pix2pix.py`](./src/pix2pix.py) script. The generator and discriminator models are defined in the `UNet` and `PatchGAN` functions respectively. The training loop is defined in the `fit` function. The loss functions are defined in the `g_loss` and `d_loss` functions for the generator and discriminator respectively.

### Loss Functions

#### Generator Loss

The generator loss function is implemented as the sum of binary crossentropy between the output of the discriminator when presented with the generated image and the real class (tensor of ones) and scaled L1 loss between the generated image and the target image. The L1 loss is scaled by a factor of 100 as per the pix2pix paper. The loss function is defined in the `g_loss` function in the [`pix2pix.py`](./src/pix2pix.py) script like so:

```py
def g_loss(
    l1_lambda: float,
    fake_G_image: tf.Tensor,
    fake_D_out: tf.Tensor,
    y: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    real_class = tf.ones_like(fake_D_out)

    fake_D_loss = bce(real_class, fake_D_out)
    l1_loss = l1(fake_G_image, y)

    return fake_D_loss + l1_lambda * l1_loss, fake_D_loss, l1_loss
```

The binary crossentropy loss is calculated as mentioned because the generator is trying to fool the discriminator into thinking that the generated image is real. The L1 loss is calculated as mentioned because the generator is trying to minimize the L1 distance between the generated image and the target image.

#### Discriminator Loss

The discriminator loss function is implemented as the sum of binary crossentropy between the output of the discriminator when presented with the real image and the real class (tensor of ones) and the output of the discriminator when presented with the generated image and the fake class (tensor of zeros). The loss function is defined in the `d_loss` function in the [`pix2pix.py`](./src/pix2pix.py) script like so:

```py
def d_loss(
    real_D_out: tf.Tensor, fake_D_out: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    real_class = tf.ones_like(real_D_out)
    fake_class = tf.zeros_like(fake_D_out)

    real_loss = bce(real_class, real_D_out)
    fake_loss = bce(fake_class, fake_D_out)

    return real_loss + fake_loss, real_loss, fake_loss
```

The binary crossentropy loss is calculated as mentioned because the discriminator is trying to correctly classify the real and generated images as real and fake respectively.

## Training

Training was tracked using [Weights & Biases](https://docs.wandb.ai/) and their Python library. The tracked metrics include total and partial losses for the generator and discriminator. These include all values returned by the loss functions defined above.

The following code snippet is a dictionary of all the tracked metrics:

```py
losses = {
    "total_gen_loss": total_gen_loss,
    "fake_d_real_class_loss": fake_d_real_class_loss,
    "l1_loss": l1_loss,
    "total_disc_loss": total_disc_loss,
    "real_d_loss": real_d_loss,
    "fake_d_loss": fake_d_loss,
}
```

where

- `total_gen_loss` is the total generator loss
- `fake_d_real_class_loss` is the binary crossentropy loss between the output of the discriminator when presented with the generated image and the real class (tensor of ones)
- `l1_loss` is the unscaled L1 loss between the generated image and the target ground truth image
- `total_disc_loss` is the total discriminator loss
- `real_d_loss` is the binary crossentropy loss between the output of the discriminator when presented with the real image and the real class (tensor of ones)
- `fake_d_loss` is the binary crossentropy loss between the output of the discriminator when presented with the generated image and the fake class (tensor of zeros)

We also generated an image after every epoch from the validation dataset and logged it to Weights & Biases along with the input image and ground truth image. This was done to be able to visually inspect the quality of the generated images so the number of images generated corresponded with the number of epochs. The validation set was also used for model testing after training was complete - we can afford to do this because GAN models are not validated like other models that can be early stopped based on simple validation metrics and because image generation during training was not exhaustive of the validation set. Apart from metrics, we also saved models in h5 format every 10 epochs. This was done to be able to resume training from a checkpoint if the training process was interrupted.

The report from Weights & Biases for each training containing logged metrics and generated images can be found [here](https://api.wandb.ai/links/nsiete23/39ioq8dy). Some of the trainings do not have corresponding notebooks in the [`src`](./src/) directory because models from those trainings were not used in the final evaluation.

**Note:**

Please note that in the first 6 trainings there was a mistake in our implementation of the PatchGAN model which affected the receptive field sizes. This means that the receptive field sizes mentioned in the following 6 sections are incorrect. The receptive sizes were 22, 94, and 382 instead of 16, 70, and 286 respectively. The mistake was fixed before the 7th training.

### Training 1

The first training was set to 200 epochs and the receptive field size in the PatchGAN model was set to 70. We used the dataset pipeline that includes the `extract_patches` step. Our free GPU runtime on Kaggle got exhausted after 172 epochs on one of our 2 accounts. Thanks to the saved models, we were able to resume training from the last checkpoint on epoch 170 on a different Kaggle account. We decided to continue training the model for 80 more epochs making the total number of epochs 250 instead of 200. The training process finished successfully on the second account and took around 17 hours in total to complete. Some of the images generated during training lacked more detail than others and some of them were somewhat washed out.

### Training 2

In the next training, the batch size was increased from 1 to 4. Other than that, the training was the same as the previous one. This training took 6 hours and 47 minutes to complete. The larger batch size did not improve the quality of the generated images during training.

The notebook for this training can be found in [`src/pix2pix-b4-mistake-in-impl.ipynb`](./src/pix2pix-b4-mistake-in-impl.ipynb).

### Training 3

The following training was set to 150 epochs and batch size was increased from 4 to 10. Other hyperparameters were the same as the previous training. This training took 3 hours and 17 minutes to complete. The larger batch size caused the resulting generated images during training to lack structure. The model was mostly incapable of generating images with straight lines and buildings looked like they were melting into their surroundings.

The notebook for this training can be found in [`src/pix2pix-b10-mistake-in-impl.ipynb`](./src/pix2pix-b10-mistake-in-impl.ipynb).

### Training 4

Another training was set to 200 epochs and batch size was set to 1 again. However, we skipped the `extract_patches` step and instead we just resized the images to 256x256 using the `resize_images` function. The training crashed after 160 epoch due to exceeding the RAM size on Kaggle because we increased the buffer size for shuffling to 1096. We continued from the last checkpoint on epoch 160. After the next 20 epochs the training crashed again so we had to decrease the buffer size to 256 to continue training for 20 more epochs. In total, this training took 4 hours and 35 minutes to complete. By only resizing the images, the generated images suffered from lesser detail present in the training images.

### Training 5

Next, we trained the model for 200 epochs with batch size set to 1. We skipped the `extract_patches` and used the `resize_images` function instead again. The PatchGAN model's patch size was set to 16 instead of 70. This led to a significant decrease in generated image quality as there were artifacts caused by small the receptive field of the discriminator. For this reason, we cancelled the training after 79 epochs. The artifacts can be seen in the generated image below.

![Generated image from epoch 79](./figures/patch_size_16_whole_map.png)

The following image is the generated image after the first epoch where we can clearly see the impact of the size of the receptive field of the discriminator.

![Generated image from epoch 1](./figures/patch_size_16_whole_map_e1.png)

### Training 6

In the next training, we tried the largest receptive field size for the PatchGAN model which is 286. We trained the model for 200 epochs with batch size set to 1. We used the `extract_patches` function again as the generator was able to generate images with more detail from zoomed in maps - resizing the original images instead of extracting multiple smaller patches from them led to a loss of detail. After 12 hours the training got interrupted as Kaggle only allows 12 hour sessions. We continued from the last checkpoint on epoch 140. Due to an unknown issue Kaggle crashed and the training got interrupted after 29 more epochs so we had to continue from the last checkpoint on epoch 160. The training lasted 17 hours and 38 minutes in total. The larger receptive field size seemed to have a positive impact on the generated images, judging by the generated validation images during training.

**Note:**

After these trainings we discovered a mistake in our implementation of the PatchGAN model. We were following the pix2pix paper which mentioned that every convolution is strided with a stride of 2. However, the [authors' implementation](https://github.com/phillipi/pix2pix/blob/master/models.lua) in the Torch framework for the Lua language used a stride of 1 for the second to last convolution layer. This means that the receptive field size of our discriminator was 22 instead of 16, 94 instead of 70 or 382 instead of 286.

### Training 7

We fixed the receptive field size issue and tried training the model again with the same hyperparameters as in the paper. We didn't use our `extract_patches` function. Instead, we only resized the images to 256x256 as in the paper. We trained the model for 200 epochs with batch size set to 1 and receptive field size of the discriminator set to 70. This training furter confirmed that not using `extract_patches` leads to a loss of detail in the generated images. The training lasted 4 hours and 6 minutes.

We decided to continue training the model for 200 more epochs to see if more training would lead to better results. We continued from the last checkpoint on epoch 200. The training lasted 4 hours and 9 minutes. This led to more detail in the generated images, however, the images still lacked detail compared to the images generated by the model trained with `extract_patches`.

The notebook for this training can be found in [`src/pix2pix-b1-rf70-e400-resize.ipynb`](./src/pix2pix-b1-rf70-e400-resize.ipynb).

### Training 8

Since the previous trainings confirmed that using `extract_patches` leads to better results, we continued using it in the next training. To confirm that the larger receptive field size has a positive impact on the generated images after fixing the receptive field size issue, we trained the model for 200 epochs with batch size set to 1 and the PatchGAN model's patch size to 286. The generated images during training looked more detailed than the images that were only resized and the largest patch size led to the best results so far. The training lasted 16 hours and 55 minutes.

The notebook for this training can be found in [`src/pix2pix-b1-rf286-e200-patches.ipynb`](./src/pix2pix-b1-rf286-e200-patches.ipynb).

### Training 9

In the next training, we set the PatchGAN model's patch size to 70. We trained the model for 200 epochs with batch size set to 1. We already trained the model with these hyperparameters in training 1, however, it was with the mistake in the implementation of the PatchGAN model. We used our `extract_patches` function again. The training lasted 14 hours and 55 minutes. Using the pix2pix model with the discriminator's receptive field size of 70 seemed to have produced lower quality images by a small margin during training compared to the model with the largest receptive field size. However, we will have to evaluate the models to see which one is better.

The notebook for this training can be found in [`src/pix2pix-b1-rf70-e200-patches.ipynb`](./src/pix2pix-b1-rf70-e200-patches.ipynb).

## Evaluation

The model was evaluated on the validation set of 1098 images. Apart from visually inspecting generated images, we also used a custom function to calculate the Structural Similarity Index (SSIM), Peak Signal-to-Noise ratio (PSNR), and the L1 distance (Mean Absolute Error) between the generated images and the ground truth images. In addition to that, we also calculated the L1 distance between highpass filtered generated images and highpass filtered ground truth images to see how well edges are preserved. We named this metric HP-L1.

PSNR ranges from 0 to infinity and higher values indicate better reconstruction quality. SSIM ranges from -1 to 1, however, it is usually between 0 and 1. SSIM of 1 means that the images are identical and SSIM of 0 means that there is no correlation between the images.

The highpass filter for HP-L1 was implemented using a 3x3 Laplacian kernel which is defined as a constant [`tf.Tensor`](https://www.tensorflow.org/api_docs/python/tf/Tensor) and is applied convolutionally to the images using the [`tf.nn.conv2d`](https://www.tensorflow.org/api_docs/python/tf/nn/conv2d) function with a stride of 1 and same padding. The kernel is defined as follows:

```py
tf.constant(
    [
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
    ],
    dtype=tf.float32
)
```

The evaluation function is defined in the [`pix2pix.py`](./src/pix2pix.py) script in the `evaluate` function. The function aggregates the SSIM, PSNR, L1 distance, and HP-L1 distance between generated images and ground truth images and returns the mean, minimum, maximum, and standard deviation of each metric in a dictionary. The metrics are also printed in a Rich table.

Upon visual inspection we found that the model trained with the batch size set to 10 produced lower quality images. Using a batch size of 4 wasn't optimal either. This is caused by the fact that batch normalization aggregated statistics over 10 and 4 images for batch sizes of 10 and 4 respectively. Using a batch size of 1 is an approach to batch normalization called instance normalization which yields better results.

Using PatchGAN with the smallest receptive field size led to artifacts in the generated images, therefore we do not include it in the final evaluattion. The larger receptive field sizes of 70x70 and 286x286 led to the best results. The models trained with `extract_patches` produced better results than the model trained using only resized images.

We chose to evaluate the models from [training 7](#training-7), [training 8](#training-8), and [training 9](#training-9) with the batch size set to 1 and corresponding PatchGAN receptive field sizes of 70, 286, and 70 respectively. We also include results from [training 2](#training-2) and [training 3](#training-3) with the batch size set to 4 and 10 respectively and discriminator receptive field size of 94 which was caused by the mistake in the implementation of the PatchGAN model.

The resulting numbers show that quality perceived by the human eye is not always reflected in metrics. The metrics are shown in a table for each model. We also include generated image samples from each model.

The notebook for the evaluation can be found in [`src/pix2pix-eval.ipynb`](./src/pix2pix-eval.ipynb).

### Model from training 7

Batch size: 1\
PatchGAN receptive field size: 70\
Epochs: 400\
Images: resized only

| Metric | Mean    | Minimum | Maximum | Standard deviation |
| ---    | ---     | ---     | ---     | ---                |
| SSIM   | 0.1117  | 0.0256  | 0.8711  | 0.1124             |
| PSNR   | 14.2976 | 10.7300 | 27.7124 | 2.1979             |
| L1     | 0.2992  | 0.0716  | 0.4493  | 0.0630             |
| HP-L1  | 0.1258  | 0.0308  | 0.1767  | 0.0247             |

**Sample generated images:**

We can see that the generated images with a similar map pattern contain structures at the same locations at the bottom of the image - 4 identical structures next to each other. This type of behavior is caused by the fact that the model learned to repeat these structures in similar maps. The repetition can be especially seen in the first 3 images of the collapsible '*Show more*' section.

![Sample 1 - Training 7](./figures/training_7/t7_0.png)

![Sample 2 - Training 7](./figures/training_7/t7_1.png)

<details>
  <summary>Show more</summary>

![Sample 3 - Training 7](./figures/training_7/t7_2.png)

![Sample 4 - Training 7](./figures/training_7/t7_3.png)

![Sample 5 - Training 7](./figures/training_7/t7_4.png)

![Sample 6 - Training 7](./figures/training_7/t7_5.png)

![Sample 7 - Training 7](./figures/training_7/t7_6.png)

![Sample 8 - Training 7](./figures/training_7/t7_7.png)

</details>

### Model from training 8

Batch size: 1\
PatchGAN receptive field size: 286\
Epochs: 200\
Images: preprocessed with `extract_patches`

| Metric | Mean    | Minimum | Maximum | Standard deviation |
| ---    | ---     | ---     | ---     | ---                |
| SSIM   | 0.1656  | -0.0550 | 0.9602  | 0.1356             |
| PSNR   | 14.5018 | 8.0866  | 34.5621 | 2.7823             |
| L1     | 0.2977  | 0.0290  | 0.7322  | 0.0722             |
| HP-L1  | 0.1136  | 0.0134  | 0.1728  | 0.0254             |

**Sample generated images:**

These images seem to be the 2nd best results based on visual inspection. The images mostly contain straight structures and the generated images are similar to the ground truth images in terms of the map pattern and amount of greenery.

![Sample 1 - Training 8](./figures/training_8/t8_0.png)

![Sample 2 - Training 8](./figures/training_8/t8_1.png)

<details>
  <summary>Show more</summary>

![Sample 3 - Training 8](./figures/training_8/t8_2.png)

![Sample 4 - Training 8](./figures/training_8/t8_3.png)

![Sample 5 - Training 8](./figures/training_8/t8_4.png)

![Sample 6 - Training 8](./figures/training_8/t8_5.png)

![Sample 7 - Training 8](./figures/training_8/t8_6.png)

![Sample 8 - Training 8](./figures/training_8/t8_7.png)

</details>

### Model from training 9

Batch size: 1\
PatchGAN receptive field size: 70\
Epochs: 200\
Images: preprocessed with `extract_patches`

| Metric | Mean    | Minimum | Maximum | Standard deviation |
| ---    | ---     | ---     | ---     | ---                |
| SSIM   | 0.1576  | -0.0621 | 0.9641  | 0.1427             |
| PSNR   | 14.1460 | 8.6875  | 33.6759 | 2.9903             |
| L1     | 0.3114  | 0.0299  | 0.6231  | 0.0805             |
| HP-L1  | 0.1108  | 0.0111  | 0.1648  | 0.0261             |

**Sample generated images:**

These images seem to be the best results from all the models. The images contain structures that are similar to real buildings and greenery.

![Sample 1 - Training 9](./figures/training_9/t9_0.png)

![Sample 2 - Training 9](./figures/training_9/t9_1.png)

<details>
  <summary>Show more</summary>

![Sample 3 - Training 9](./figures/training_9/t9_2.png)

![Sample 4 - Training 9](./figures/training_9/t9_3.png)

![Sample 5 - Training 9](./figures/training_9/t9_4.png)

![Sample 6 - Training 9](./figures/training_9/t9_5.png)

![Sample 7 - Training 9](./figures/training_9/t9_6.png)

![Sample 8 - Training 9](./figures/training_9/t9_7.png)

</details>

### Model from training 2

Batch size: 4\
PatchGAN receptive field size: 94 (mistake in implementation)\
Epochs: 200\
Images: preprocessed with `extract_patches`

| Metric | Mean    | Minimum | Maximum | Standard deviation |
| ---    | ---     | ---     | ---     | ---                |
| SSIM   | 0.1592  | -0.0381 | 0.9653  | 0.1364             |
| PSNR   | 14.6817 | 8.7748  | 33.6167 | 2.8260             |
| L1     | 0.2913  | 0.0332  | 0.6202  | 0.0720             |
| HP-L1  | 0.0969  | 0.0089  | 0.1416  | 0.0225             |

**Sample generated images:**

The images show that the model has learned to generate too much greenery even in places where there is no greenery in the ground truth images. The images lack structures that resemble real buildings.

![Sample 1 - Training 2](./figures/training_2/t2_0.png)

![Sample 2 - Training 2](./figures/training_2/t2_1.png)

<details>
  <summary>Show more</summary>

![Sample 3 - Training 2](./figures/training_2/t2_2.png)

![Sample 4 - Training 2](./figures/training_2/t2_3.png)

![Sample 5 - Training 2](./figures/training_2/t2_4.png)

![Sample 6 - Training 2](./figures/training_2/t2_5.png)

![Sample 7 - Training 2](./figures/training_2/t2_6.png)

![Sample 8 - Training 2](./figures/training_2/t2_7.png)

</details>

### Model from training 3

Batch size: 10\
PatchGAN receptive field size: 94 (mistake in implementation)\
Epochs: 150\
Images: preprocessed with `extract_patches`

| Metric | Mean    | Minimum | Maximum | Standard deviation |
| ---    | ---     | ---     | ---     | ---                |
| SSIM   | 0.1549  | -0.0336 | 0.8826  | 0.1278             |
| PSNR   | 14.1051 | 7.8994  | 32.6551 | 2.8017             |
| L1     | 0.3123  | 0.0345  | 0.7272  | 0.0766             |
| HP-L1  | 0.0932  | 0.0138  | 0.1580  | 0.0266             |

**Sample generated images:**

Images generated by this model are the worst from all the models. The images fail to copy the map structure well and the streets are not defined as well as in images generated by models from training 8 and 9.

![Sample 1 - Training 3](./figures/training_3/t3_0.png)

![Sample 2 - Training 3](./figures/training_3/t3_1.png)

<details>
  <summary>Show more</summary>

![Sample 3 - Training 3](./figures/training_3/t3_2.png)

![Sample 4 - Training 3](./figures/training_3/t3_3.png)

![Sample 5 - Training 3](./figures/training_3/t3_4.png)

![Sample 6 - Training 3](./figures/training_3/t3_5.png)

![Sample 7 - Training 3](./figures/training_3/t3_6.png)

![Sample 8 - Training 3](./figures/training_3/t3_7.png)

</details>

## References

- [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004), Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, Alexei A. Efros, 2016

- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597), Olaf Ronneberger, Philipp Fischer, and Thomas Brox, 2015

- [TensorFlow pix2pix tutorial](https://www.tensorflow.org/tutorials/generative/pix2pix), The TensorFlow Authors, 2019

- [pix2pix Torch implementation](https://github.com/phillipi/pix2pix), Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, Alexei A. Efros, 2017

- [Is there a relationship between peak-signal-to-noise ratio and structural similarity index measure?](https://doi.org/10.1049/iet-ipr.2012.0489), Alain Horé, Djemel Ziou, 2013
