from typing import Dict, Tuple, Union, cast
from matplotlib.figure import Figure
import os
import numpy as np
import wandb
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy, MeanAbsoluteError
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras import Model, Sequential, Input
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras.layers import (
    Conv2D,
    Conv2DTranspose,
    BatchNormalization,
    ReLU,
    LeakyReLU,
    Dropout,
    Concatenate,
)

bce = BinaryCrossentropy()
l1 = MeanAbsoluteError()


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

    fake_D_loss = bce(real_class, fake_D_out)
    l1_loss = l1(fake_G_image, y)

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

    real_loss = bce(real_class, real_D_out)
    fake_loss = bce(fake_class, fake_D_out)

    return real_loss + fake_loss, real_loss, fake_loss


def downsampling_block(
    filters: int,
    kernel_size: Tuple[int, int] = (4, 4),
    strides: Tuple[int, int] = (2, 2),
    use_batchnorm: bool = True,
    dropout: Union[float, None] = None,
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
    dropout: Union[float, None] = None,
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
    generator: Model, example_input, example_target, show=False
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
    l1_loss = l1(example_target, prediction)
    fig = plt.figure(figsize=(10, 10))

    display_list = [example_input[0], example_target, prediction[0]]
    titles = ["Input Image", "Ground Truth", "Predicted Image"]

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        title = titles[i]
        if i == 2:
            title += f"\nL1 Loss: {l1_loss:.4f}"
        plt.title(title)
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis("off")
    if show:
        plt.show()
    else:
        plt.close()

    return fig


@tf.function
def train_step(
    generator: Model,
    discriminator: Model,
    generator_optimizer: Optimizer,
    discriminator_optimizer: Optimizer,
    l1_lambda: float,
    input_image: tf.Tensor,
    target: tf.Tensor,
) -> Dict[str, tf.Tensor]:
    """
    Performs a single training step for the pix2pix model.

    :param generator: The generator model.
    :param discriminator: The discriminator model.
    :param generator_optimizer: The optimizer for the generator.
    :param discriminator_optimizer: The optimizer for the discriminator.
    :param l1_lambda: The lambda value for the L1 loss.
    :param input_image: The input image to be translated.
    :param target: The target ground truth image.

    :return: A dictionary containing the losses for the generator and discriminator.
    """
    with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
        fake_g_image = generator(input_image, training=True)

        real_d_out = discriminator([input_image, target], training=True)
        fake_d_out = discriminator([input_image, fake_g_image], training=True)

        total_gen_loss, fake_d_real_class_loss, l1_loss = g_loss(
            l1_lambda, fake_g_image, fake_d_out, target
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
    train_data,
    val_data,
    epochs: int,
    generator: Model,
    discriminator: Model,
    generator_optimizer: Optimizer,
    discriminator_optimizer: Optimizer,
    l1_lambda: float = 100,
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
                f"Epoch: {epoch + 1}, Step: {step}, Gen Loss: {gen_loss}, Disc Loss: {disc_loss}",
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
            example_target_batch[0],
            show=(epoch + 1) % 10 == 0 or epoch == 0,
        )
        wandb.log({**losses_epoch, "epoch": epoch + 1, "image": figure})
        if (epoch + 1) % 10 == 0:
            generator.save(f"generator_{epoch + 1}.h5")
            discriminator.save(f"discriminator_{epoch + 1}.h5")
            wandb.save(f"generator_{epoch + 1}.h5")
            wandb.save(f"discriminator_{epoch + 1}.h5")


if __name__ == "__main__":
    gen = UNet((256, 256, 3))
    gen.summary()
    plot_model(gen, to_file="generator.png", show_shapes=True)

    disc = PatchGAN((256, 256, 3))
    disc.summary()
    plot_model(disc, to_file="discriminator.png", show_shapes=True)
