# pix2pix for Maps to Satellite Images Translation

## Dataset

The dataset used for this project is the paired [Pix2pix Maps](http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/maps.tar.gz) dataset containing 1096 training images and 1098 validation images. The dataset contains images of maps and their corresponding satellite images extracted from Google Maps. The dataset is also available on Kaggle [here](https://www.kaggle.com/datasets/alincijov/pix2pix-maps).

The map to satellite pairs are in one jpeg file, side-by-side. The size of the images is 600x1200. For this reason, the images need to be split into two separate images. We defined a custom `tf.data.Dataset` pipeline to load the images and split them into two separate images. This step is done after loading the image file using a function called `load_image` which is passed to the `map` function of the dataset pipeline. The `load_image` function is defined as follows:

```py
def load_image(image_file):
    image = tf.io.read_file(image_file)
    image = tf.io.decode_jpeg(image)
    
    w = tf.shape(image)[1]
    w = w // 2
    
    input_image = image[:, w:, :]
    real_image = image[:, :w, :]
    
    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)
    
    return input_image, real_image
```
