from typing import Dict, Optional, Tuple, cast
from matplotlib.figure import Figure
import os
import wandb
import matplotlib.pyplot as plt
import tensorflow as tf
from rich.table import Table
from rich.console import Console
from mpl_toolkits.axes_grid1 import ImageGrid

# Ignore warnings due to Pylance not being able to resolve imports
from tensorflow.keras.losses import BinaryCrossentropy, MeanAbsoluteError  # type: ignore
from tensorflow.keras.initializers import RandomNormal  # type: ignore
from tensorflow.keras import Model, Sequential, Input  # type: ignore
from tensorflow.keras.utils import plot_model  # type: ignore
from tensorflow.keras.optimizers import Optimizer  # type: ignore
from tensorflow.keras.layers import (  # type: ignore
    Conv2D,
    Conv2DTranspose,
    BatchNormalization,
    ReLU,
    LeakyReLU,
    Dropout,
    Concatenate,
)

bce_object = BinaryCrossentropy()
l1_object = MeanAbsoluteError()


def g_loss(
    l1_lambda: float,
    fake_G_image: tf.Tensor,
    fake_D_out: tf.Tensor,
    y: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Calculates the generator loss.

    :param l1_lambda: The lambda value for the L1 loss.
    :param fake_G_image: The fake generated image.
    :param fake_D_out: The fake discriminator output.
    :param y: The real image.

    :return: The total generator loss, the loss between the fake
    discriminator output and real class, and the L1 loss
    between the fake generated image and the real image.
    """
    real_class = tf.ones_like(fake_D_out)

    fake_D_loss = bce_object(real_class, fake_D_out)
    l1_loss = l1_object(fake_G_image, y)

    return fake_D_loss + l1_lambda * l1_loss, fake_D_loss, l1_loss


def d_loss(
    real_D_out: tf.Tensor, fake_D_out: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Calculates the discriminator loss.

    :param real_D_out: The real discriminator output.
    :param fake_D_out: The fake discriminator output.

    :return: The total discriminator loss, the discriminator loss
    between the real discriminator output and real class, and the
    discriminator loss between the fake discriminator output and
    fake class.
    """
    real_class = tf.ones_like(real_D_out)
    fake_class = tf.zeros_like(fake_D_out)

    real_loss = bce_object(real_class, real_D_out)
    fake_loss = bce_object(fake_class, fake_D_out)

    return real_loss + fake_loss, real_loss, fake_loss


def downsampling_block(
    filters: int,
    kernel_size: Tuple[int, int] = (4, 4),
    strides: Tuple[int, int] = (2, 2),
    use_batchnorm: bool = True,
    dropout: Optional[float] = None,
) -> Sequential:
    """
    Creates a downsampling block.

    :param filters: The number of filters.
    :param kernel_size: The kernel size.
    :param strides: The strides.
    :param use_batchnorm: Whether to use batch normalization.
    :param dropout: The dropout rate. If None, no dropout is used.

    :return: The downsampling block.
    """
    down = Sequential()

    down.add(
        Conv2D(
            filters,
            kernel_size=kernel_size,
            strides=strides,
            padding="same",
            kernel_initializer=RandomNormal(mean=0.0, stddev=0.02),
        )
    )
    if use_batchnorm:
        down.add(BatchNormalization())
    if dropout:
        down.add(Dropout(dropout))
    down.add(LeakyReLU(0.2))

    return down


def upsampling_block(
    filters: int,
    kernel_size: Tuple[int, int] = (4, 4),
    strides: Tuple[int, int] = (2, 2),
    use_batchnorm: bool = True,
    dropout: Optional[float] = None,
) -> Sequential:
    """
    Creates an upsampling block.

    :param filters: The number of filters.
    :param kernel_size: The kernel size.
    :param strides: The strides.
    :param use_batchnorm: Whether to use batch normalization.
    :param dropout: The dropout rate. If None, no dropout is used.

    :return: The upsampling block.
    """
    up = Sequential()

    up.add(
        Conv2DTranspose(
            filters,
            kernel_size=kernel_size,
            strides=strides,
            padding="same",
            kernel_initializer=RandomNormal(mean=0.0, stddev=0.02),
        )
    )
    if use_batchnorm:
        up.add(BatchNormalization())
    if dropout:
        up.add(Dropout(dropout))
    up.add(ReLU())

    return up


def UNet(
    input_shape: Tuple[int, int, int],
) -> Model:
    """
    Creates a UNet generator model.

    :param input_shape: The input shape of the model.

    :return: The UNet generator model.
    """
    inputs = Input(shape=input_shape)

    contracting_path = [
        downsampling_block(64, use_batchnorm=False),
        downsampling_block(128),
        downsampling_block(256),
        downsampling_block(512),
        downsampling_block(512),
        downsampling_block(512),
        downsampling_block(512),
        downsampling_block(512),
    ]

    expansive_path = [
        upsampling_block(512, dropout=0.5),
        upsampling_block(512, dropout=0.5),
        upsampling_block(512, dropout=0.5),
        upsampling_block(512),
        upsampling_block(256),
        upsampling_block(128),
        upsampling_block(64),
    ]

    last = Conv2DTranspose(
        3,
        kernel_size=(4, 4),
        strides=(2, 2),
        activation="tanh",
        padding="same",
        kernel_initializer=RandomNormal(mean=0.0, stddev=0.02),
    )

    out = inputs

    skip_connections = []
    for down in contracting_path:
        out = down(out)
        skip_connections.append(out)

    for up, skip in zip(expansive_path, reversed(skip_connections[:-1])):
        out = up(out)
        out = Concatenate()([out, skip])

    out = last(out)

    return Model(inputs=inputs, outputs=out)


def PatchGAN(
    input_shape: Tuple[int, int, int],
    patch_size: int = 70,
) -> Model:
    """
    Creates a PatchGAN discriminator model.

    :param input_shape: The input shape of the model.
    :param patch_size: The size of the patches.

    :return: The PatchGAN discriminator model.
    """
    if patch_size not in (1, 16, 70, 286):
        raise ValueError("Patch size must be 1, 16, 70 or 286.")

    inputs = Input(shape=input_shape, name="input_image")
    targets = Input(shape=input_shape, name="target_image")

    out = Concatenate()([inputs, targets])

    kernel_size = (4, 4)
    if patch_size == 1:
        kernel_size = (1, 1)

    out = downsampling_block(64, kernel_size=kernel_size, use_batchnorm=False)(out)
    out = downsampling_block(128, kernel_size=kernel_size)(out)

    if patch_size not in (1, 16):
        out = downsampling_block(256, kernel_size=kernel_size)(out)
        out = downsampling_block(512, kernel_size=kernel_size)(out)

    if patch_size == 286:
        out = downsampling_block(512, kernel_size=kernel_size)(out)
        out = downsampling_block(512, kernel_size=kernel_size)(out)

    out = Conv2D(
        1,
        kernel_size=kernel_size,
        padding="same",
        activation="sigmoid",
        kernel_initializer=RandomNormal(mean=0.0, stddev=0.02),
    )(out)

    return Model(inputs=[inputs, targets], outputs=out)


def generate_image(
    generator: Model,
    example_input: tf.Tensor,
    example_target: tf.Tensor,
    show: bool = False,
) -> Figure:
    """
    Generates and optionally displays a generated image
    from the generator model along with the input and
    ground truth images.

    :param generator: The generator model
    :param example_input: The input image to be translated
    :param example_target: The ground truth image
    :param show: Whether to display the generated image

    :return: Figure of example input, target and prediction
    """
    prediction = generator(example_input, training=True)
    l1_loss = l1_object(example_target, prediction)
    fig = plt.figure(figsize=(15, 5))

    display_list = [example_input[0], example_target[0], prediction[0]]  # type: ignore
    titles = ["Input Image", "Ground Truth", "Predicted Image"]

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        title = titles[i]
        if i == 2:
            title += f"\nL1 Loss: {l1_loss:.4f}"
        plt.title(title)
        plt.imshow(tf.add(tf.multiply(display_list[i], 0.5), 0.5))
        plt.axis("off")
    if show:
        plt.show()
    else:
        plt.close()

    return fig


def show_generated_image(
    input_image: tf.Tensor,
    target_image: tf.Tensor,
    predicted_image: tf.Tensor,
) -> None:
    """
    Displays a generated image from the generator model.

    :param input_image: The input image.
    :param target_image: The ground truth image.
    :param predicted_image: The predicted image.

    :return: None
    """
    plt.figure(figsize=(15, 5))

    display_list = [input_image, target_image, predicted_image]
    titles = ["Input Image", "Ground Truth", "Predicted Image"]

    l1_loss = l1_object(target_image, predicted_image)

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        title = titles[i]
        if i == 2:
            title += f"\nL1 Loss: {l1_loss:.4f}"
        plt.title(title)
        plt.imshow(tf.add(tf.multiply(display_list[i], 0.5), 0.5))
        plt.axis("off")

    plt.show()


@tf.function
def train_step(
    generator: Model,
    discriminator: Model,
    generator_optimizer: Optimizer,
    discriminator_optimizer: Optimizer,
    l1_lambda: float,
    input_batch: tf.Tensor,
    target_batch: tf.Tensor,
) -> Dict[str, tf.Tensor]:
    """
    Performs a single training step for the pix2pix model.

    :param generator: The generator model.
    :param discriminator: The discriminator model.
    :param generator_optimizer: The optimizer for the generator.
    :param discriminator_optimizer: The optimizer for the discriminator.
    :param l1_lambda: The lambda value for the L1 loss.
    :param input_batch: The input image batch to be translated.
    :param target_batch: The ground truth image batch.

    :return: A dictionary containing the losses for the generator and discriminator.
    """
    with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
        fake_g_image = generator(input_batch, training=True)

        real_d_out = discriminator([input_batch, target_batch], training=True)
        fake_d_out = discriminator([input_batch, fake_g_image], training=True)

        total_gen_loss, fake_d_real_class_loss, l1_loss = g_loss(
            l1_lambda, fake_g_image, fake_d_out, target_batch
        )
        total_disc_loss, real_d_loss, fake_d_loss = d_loss(real_d_out, fake_d_out)

    generator_gradients = g_tape.gradient(total_gen_loss, generator.trainable_variables)
    discriminator_gradients = d_tape.gradient(
        total_disc_loss, discriminator.trainable_variables
    )

    generator_optimizer.apply_gradients(
        zip(generator_gradients, generator.trainable_variables)
    )
    discriminator_optimizer.apply_gradients(
        zip(discriminator_gradients, discriminator.trainable_variables)
    )

    losses = {
        "total_gen_loss": total_gen_loss,
        "fake_d_real_class_loss": fake_d_real_class_loss,
        "l1_loss": l1_loss,
        "total_disc_loss": total_disc_loss,
        "real_d_loss": real_d_loss,
        "fake_d_loss": fake_d_loss,
    }

    return losses


def fit(
    train_data: tf.data.Dataset,
    val_data: tf.data.Dataset,
    epochs: int,
    generator: Model,
    discriminator: Model,
    generator_optimizer: Optimizer,
    discriminator_optimizer: Optimizer,
    l1_lambda: float = 100,
    use_wandb: bool = True,
    save_checkpoints: bool = True,
) -> None:
    """
    Trains the pix2pix model.

    :param train_data: Training data.
    :param val_data: Validation data.
    :param epochs: Number of epochs to train for.
    :param generator: Generator model.
    :param discriminator: Discriminator model.
    :param generator_optimizer: Generator optimizer.
    :param discriminator_optimizer: Discriminator optimizer.
    :param l1_lambda: Lambda for L1 loss.

    :return: None.
    """
    for epoch in range(epochs):
        losses_epoch = {}
        for step, (input_batch, target_batch) in enumerate(train_data):
            losses = train_step(
                generator,
                discriminator,
                generator_optimizer,
                discriminator_optimizer,
                l1_lambda,
                input_batch,
                target_batch,
            )

            losses = cast(Dict[str, tf.Tensor], losses)
            gen_loss = losses["total_gen_loss"]
            disc_loss = losses["total_disc_loss"]

            losses_epoch = {k: losses_epoch.get(k, []) + [v] for k, v in losses.items()}

            print(
                f"Epoch: {epoch + 1}, Step: {step + 1}, Gen Loss: {gen_loss:.4f}, Disc Loss: {disc_loss:.4f}",
                end="\r",
                flush=True,
            )
        print("")

        for k, v in losses_epoch.items():
            losses_epoch[k] = tf.reduce_mean(v)

        example_input_batch, example_target_batch = next(iter(val_data.take(1)))
        figure = generate_image(
            generator,
            tf.expand_dims(example_input_batch[0], axis=0),
            tf.expand_dims(example_target_batch[0], axis=0),
            show=(epoch + 1) % 10 == 0 or epoch == 0,
        )
        if use_wandb:
            wandb.log({**losses_epoch, "epoch": epoch + 1, "image": figure})
        if save_checkpoints and (epoch + 1) % 10 == 0:
            gen_path = os.path.join("checkpoints", f"generator_{epoch + 1}.h5")
            disc_path = os.path.join("checkpoints", f"discriminator_{epoch + 1}.h5")
            generator.save(gen_path)
            discriminator.save(disc_path)
            if use_wandb:
                wandb.save(gen_path)
                wandb.save(disc_path)


@tf.function
def ssim_rgb(
    y_true_batch: tf.Tensor, y_pred_batch: tf.Tensor, max_val: float = 1.0
) -> tf.Tensor:
    """
    Computes the structural similarity index between two batches
    of RGB images for each channel resulting in a weighted average.

    :param y_true_batch: The ground truth RGB image batch.
    :param y_pred_batch: The predicted RGB image batch.
    :param max_val: The maximum value of the RGB images.

    :return: The weighted average of the SSIM values for each channel
    in each batch.
    """
    return tf.reduce_mean(
        [
            tf.image.ssim(
                tf.expand_dims(y_true_batch[:, :, :, i], axis=-1),  # type: ignore
                tf.expand_dims(y_pred_batch[:, :, :, i], axis=-1),  # type: ignore
                max_val=max_val,
            )
            for i in range(3)
        ]
    )


@tf.function
def psnr_rgb(
    y_true_batch: tf.Tensor, y_pred_batch: tf.Tensor, max_val: float = 1.0
) -> tf.Tensor:
    """
    Computes the average peak signal-to-noise ratio between
    two batches of RGB images.

    :param y_true_batch: The ground truth RGB image batch.
    :param y_pred_batch: The predicted RGB image batch.
    :param max_val: The maximum value of the RGB images.

    :return: The average of the PSNR values in each batch.
    """
    return tf.reduce_mean(
        tf.image.psnr(
            y_true_batch,
            y_pred_batch,
            max_val=max_val,
        )
    )


def evaluate(
    generator: Model,
    dataset: tf.data.Dataset,
    n_samples: Optional[int] = None,
) -> Dict[str, tf.Tensor]:
    """
    Evaluates the pix2pix model on a dataset. The resulting evaluation contains
    the mean, minimum, maximum and standard deviation of the SSIM, PSNR and L1
    values over batches in the dataset.

    :param generator: Generator model to evaluate.
    :param dataset: Batched dataset to evaluate the model on.
    :param n_samples: Number of generated samples to show. If None, no
    samples are shown. Only one sample is taken from one batch. If the
    number of batches is less than n_samples, all batches are used.

    :return: A dictionary containing the mean, minimum, maximum and standard
    deviation of the SSIM, PSNR and L1 values over batches in the dataset.
    """
    ssim = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    psnr = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    l1 = tf.TensorArray(tf.float32, size=0, dynamic_size=True)

    sample_outputs = tf.TensorArray(tf.float32, size=0, dynamic_size=True)

    print("Evaluation started...")

    for batch_index, (input_batch, target_batch) in enumerate(dataset):
        # training is set to True as per the pix2pix paper to use dropout
        # and batch normalization using test time statistics
        predictions = generator(input_batch, training=True)

        if n_samples is not None and batch_index < n_samples:
            sample_outputs = sample_outputs.write(
                batch_index, tf.stack([input_batch[0], target_batch[0], predictions[0]])
            )

        ssim_step = ssim_rgb(target_batch, predictions)
        psnr_step = psnr_rgb(target_batch, predictions)
        l1_step = l1_object(target_batch, predictions)

        ssim = ssim.write(batch_index, ssim_step)
        psnr = psnr.write(batch_index, psnr_step)
        l1 = l1.write(batch_index, l1_step)

        print(
            f"Step {batch_index + 1} - SSIM: {ssim_step:.4f}, PSNR: {psnr_step:.4f}, L1: {l1_step:.4f}",
            end="\r",
            flush=True,
        )
    print("")

    ssim_tensor = ssim.stack()
    psnr_tensor = psnr.stack()
    l1_tensor = l1.stack()

    ssim_mean = tf.reduce_mean(ssim_tensor)
    ssim_min = tf.reduce_min(ssim_tensor)
    ssim_max = tf.reduce_max(ssim_tensor)
    ssim_std = tf.math.reduce_std(ssim_tensor)

    psnr_mean = tf.reduce_mean(psnr_tensor)
    psnr_min = tf.reduce_min(psnr_tensor)
    psnr_max = tf.reduce_max(psnr_tensor)
    psnr_std = tf.math.reduce_std(psnr_tensor)

    l1_mean = tf.reduce_mean(l1_tensor)
    l1_min = tf.reduce_min(l1_tensor)
    l1_max = tf.reduce_max(l1_tensor)
    l1_std = tf.math.reduce_std(l1_tensor)

    table = Table("Metric", "Mean", "Min", "Max", "Std", title="Evaluation Results")

    table.add_row(
        "SSIM",
        f"{ssim_mean:.4f}",
        f"{ssim_min:.4f}",
        f"{ssim_max:.4f}",
        f"{ssim_std:.4f}",
    )
    table.add_row(
        "PSNR",
        f"{psnr_mean:.4f}",
        f"{psnr_min:.4f}",
        f"{psnr_max:.4f}",
        f"{psnr_std:.4f}",
    )
    table.add_row(
        "L1",
        f"{l1_mean:.4f}",
        f"{l1_min:.4f}",
        f"{l1_max:.4f}",
        f"{l1_std:.4f}",
    )

    console = Console()
    console.print(table)

    if n_samples is not None:
        for input_image, target_image, predicted_image in sample_outputs.stack():
            show_generated_image(input_image, target_image, predicted_image)

    return {
        "ssim_mean": ssim_mean,
        "ssim_min": ssim_min,
        "ssim_max": ssim_max,
        "ssim_std": ssim_std,
        "psnr_mean": psnr_mean,
        "psnr_min": psnr_min,
        "psnr_max": psnr_max,
        "psnr_std": psnr_std,
        "l1_mean": l1_mean,
        "l1_min": l1_min,
        "l1_max": l1_max,
        "l1_std": l1_std,
    }


if __name__ == "__main__":
    gen = UNet((256, 256, 3))
    gen.summary()
    plot_model(gen, to_file="generator.png", show_shapes=True)

    disc = PatchGAN((256, 256, 3))
    disc.summary()
    plot_model(disc, to_file="discriminator.png", show_shapes=True)

    generate_image(gen, tf.zeros((1, 256, 256, 3)), tf.zeros((256, 256, 3)), show=True)
