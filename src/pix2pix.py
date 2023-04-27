from typing import Dict, Tuple, Union, cast
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
    fake_G: tf.Tensor,
    fake_D: tf.Tensor,
    y: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Calculates the generator loss.

    :param l1_lambda: The lambda value for the L1 loss.
    :param fake_G: The fake generated image.
    :param fake_D: The fake discriminator output.
    :param y: The real image.

    :return: The generator loss, the fake loss, and the L1 loss.
    """
    fake_loss = bce(tf.ones_like(fake_D), fake_D)
    l1_loss = l1(fake_G, y)
    return fake_loss + l1_lambda * l1_loss, fake_loss, l1_loss


def d_loss(
    real_D: tf.Tensor, fake_D: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Calculates the discriminator loss.

    :param real_D: The real discriminator output.
    :param fake_D: The fake discriminator output.

    :return: The discriminator loss, the real discriminator loss, and the fake discriminator loss.
    """
    real_loss = bce(tf.ones_like(real_D), real_D)
    fake_loss = bce(tf.zeros_like(fake_D), fake_D)
    return real_loss + fake_loss, real_loss, fake_loss


def downsampling_block(
    filters: int,
    kernel_size: Tuple[int, int] = (4, 4),
    strides: Tuple[int, int] = (2, 2),
    use_batchnorm: bool = True,
    dropout: Union[float, None] = None,
) -> Sequential:
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
    with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
        fake_G = generator(input_image, training=True)

        real_D = discriminator([input_image, target], training=True)
        fake_D = discriminator([input_image, fake_G], training=True)

        gen_loss, fake_loss, l1_loss = g_loss(l1_lambda, fake_G, fake_D, target)
        disc_loss, real_d_loss, fake_d_loss = d_loss(real_D, fake_D)

    generator_gradients = g_tape.gradient(gen_loss, generator.trainable_variables)
    discriminator_gradients = d_tape.gradient(
        disc_loss, discriminator.trainable_variables
    )

    generator_optimizer.apply_gradients(
        zip(generator_gradients, generator.trainable_variables)
    )
    discriminator_optimizer.apply_gradients(
        zip(discriminator_gradients, discriminator.trainable_variables)
    )

    losses = {
        "gen_loss": gen_loss,
        "fake_loss": fake_loss,
        "l1_loss": l1_loss,
        "disc_loss": disc_loss,
        "real_d_loss": real_d_loss,
        "fake_d_loss": fake_d_loss,
    }

    return losses


def generate_image(generator: Model, example_input, example_target):
    prediction = generator(example_input, training=True)
    plt.figure(figsize=(10, 10))

    display_list = [example_input[0], example_target[0], prediction[0]]
    title = ["Input Image", "Ground Truth", "Predicted Image"]

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(title[i])
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis("off")
    plt.show()


def fit(
    train_data,
    val_data,
    epochs: int,
    generator: Model,
    discriminator: Model,
    generator_optimizer: Optimizer,
    discriminator_optimizer: Optimizer,
    l1_lambda: float,
) -> None:
    for epoch in range(epochs):
        losses: Union[Dict[str, tf.Tensor], None] = {}
        example_input, example_target = next(iter(val_data.take(1)))
        for step, (input_image, target) in enumerate(train_data):
            losses = train_step(
                generator,
                discriminator,
                generator_optimizer,
                discriminator_optimizer,
                l1_lambda,
                input_image,
                target,
            )

            losses = cast(Dict[str, tf.Tensor], losses)
            gen_loss = losses["gen_loss"]
            disc_loss = losses["disc_loss"]

            print(
                f"Epoch: {epoch + 1}, Step: {step}, Gen Loss: {gen_loss}, Disc Loss: {disc_loss}",
                end="\r",
                flush=True,
            )
        print("\n")

        wandb.log({**losses, "epoch": epoch + 1})
        generate_image(generator, example_input, example_target)


if __name__ == "__main__":
    gen = UNet((256, 256, 3))
    gen.summary()
    plot_model(gen, to_file="generator.png", show_shapes=True)

    disc = PatchGAN((256, 256, 3))
    disc.summary()
    plot_model(disc, to_file="discriminator.png", show_shapes=True)
