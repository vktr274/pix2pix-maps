import tensorflow as tf
import matplotlib.pyplot as plt


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


def resize_images(input_image, real_image, resize_to: int):
    input_image = tf.image.resize(input_image, (resize_to, resize_to))
    real_image = tf.image.resize(real_image, (resize_to, resize_to))

    return input_image, real_image


def rescale_images(input_image, real_image):
    input_image /= 255.0
    real_image /= 255.0

    return input_image, real_image


def extract_patches(input_image, real_image, patch_size: int, num_of_patches: int):
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
    _, axs = plt.subplots(1, 2)
    axs[0].axis("off")
    axs[1].axis("off")
    axs[0].set_title(f"Input Image {number} ({subset})")
    axs[1].set_title(f"Real Image {number} ({subset})")
    axs[0].imshow(input_image)
    axs[1].imshow(real_image)
