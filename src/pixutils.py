from typing import Tuple, cast
import tensorflow as tf
import matplotlib.pyplot as plt


def load_image(image_file: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Loads an image from a file and splits it into two images.

    :param image_file string tensor of the path

    :return left and right half of the image
    """
    image = tf.io.read_file(image_file)
    image = tf.io.decode_jpeg(image)

    w = tf.shape(image)[1]
    w = w // 2

    input_image = image[:, w:, :]
    real_image = image[:, :w, :]

    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)

    return cast(Tuple[tf.Tensor, tf.Tensor], (input_image, real_image))


def resize_images(
    input_image: tf.Tensor, real_image: tf.Tensor, resize_to: int
) -> Tuple[tf.Tensor, tf.Tensor]:
    input_image = tf.image.resize(input_image, (resize_to, resize_to))
    real_image = tf.image.resize(real_image, (resize_to, resize_to))

    return input_image, real_image


def rescale_images(
    input_image: tf.Tensor, real_image: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor]:
    input_image = tf.subtract(tf.divide(input_image, 127.5), 1.0)
    real_image = tf.subtract(tf.divide(real_image, 127.5), 1.0)

    return input_image, real_image


@tf.function
def random_jitter(
    input_image: tf.Tensor, real_image: tf.Tensor, resize_to: int
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    First resizes images to the given size, then randomly crops them to the original size.
    After that, the images are randomly horizontally flipped.

    :param input_image: input image
    :param real_image: real image
    :param resize_to: size to resize the images to

    :return: randomly jittered images
    """
    original_size = tf.shape(input_image)[0]

    input_image, real_image = resize_images(
        input_image=input_image,
        real_image=real_image,
        resize_to=resize_to,
    )

    input_image = tf.image.random_crop(
        input_image,
        size=(original_size, original_size, 3),
    )
    real_image = tf.image.random_crop(
        real_image,
        size=(original_size, original_size, 3),
    )

    if tf.random.uniform(()) > 0.5:
        input_image = cast(tf.Tensor, tf.image.flip_left_right(input_image))
        real_image = cast(tf.Tensor, tf.image.flip_left_right(real_image))

    return cast(Tuple[tf.Tensor, tf.Tensor], (input_image, real_image))


@tf.function
def extract_patches(
    input_image: tf.Tensor, real_image: tf.Tensor, patch_size: int, num_of_patches: int
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Extracts patches from the given images.

    :param input_image: input image
    :param real_image: real image
    :param patch_size: size of the patches
    :param num_of_patches: number of patches that are going to be extracted

    :return: input and real image patches
    """
    input_image_patches = tf.image.extract_patches(
        images=tf.expand_dims(input_image, axis=0),
        sizes=[1, patch_size, patch_size, 1],
        strides=[1, patch_size, patch_size, 1],
        rates=[1, 1, 1, 1],
        padding="VALID",
    )
    real_image_patches = tf.image.extract_patches(
        images=tf.expand_dims(real_image, axis=0),
        sizes=[1, patch_size, patch_size, 1],
        strides=[1, patch_size, patch_size, 1],
        rates=[1, 1, 1, 1],
        padding="VALID",
    )
    input_image_patches = tf.reshape(
        input_image_patches, (num_of_patches, patch_size, patch_size, 3)
    )
    real_image_patches = tf.reshape(
        real_image_patches, (num_of_patches, patch_size, patch_size, 3)
    )
    return input_image_patches, real_image_patches


def show(input_image, real_image, number: int, subset: str) -> None:
    input_image = input_image * 0.5 + 0.5
    real_image = real_image * 0.5 + 0.5

    _, axs = plt.subplots(1, 2)
    axs[0].axis("off")  # type: ignore
    axs[1].axis("off")  # type: ignore
    axs[0].set_title(f"Input Image {number} ({subset})")  # type: ignore
    axs[1].set_title(f"Real Image {number} ({subset})")  # type: ignore
    axs[0].imshow(input_image, vmin=-1, vmax=1)  # type: ignore
    axs[1].imshow(real_image, vmin=-1, vmax=1)  # type: ignore
