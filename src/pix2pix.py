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


def generate_image(generator: Model, example_input, example_target, show=False):
    prediction = generator(example_input, training=True)
    fig = plt.figure(figsize=(10, 10))

    display_list = [example_input[0], example_target[0], prediction[0]]
    title = ["Input Image", "Ground Truth", "Predicted Image"]

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(title[i])
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis("off")
    if show:
        plt.show()
    wandb.log({"images": fig}, commit=False)


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
    l1_lambda: float,
) -> None:
    for epoch in range(epochs):
        losses_epoch = {}
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
            gen_loss = losses["total_gen_loss"]
            disc_loss = losses["total_disc_loss"]

            losses_epoch = {k: losses_epoch.get(k, []) + [v] for k, v in losses.items()}

            print(
                f"Epoch: {epoch + 1}, ",
                f"Step: {step}, ",
                f"Total Gen Loss: {gen_loss}, ",
                f"Total Disc Loss: {disc_loss}",
                end="\r",
                flush=True,
            )
        print("")

        for k, v in losses_epoch.items():
            losses_epoch[k] = tf.reduce_mean(v)

        generate_image(generator, example_input, example_target, show=epoch % 10 == 0)
        wandb.log({**losses_epoch, "epoch": epoch + 1})


if __name__ == "__main__":
    gen = UNet((256, 256, 3))
    gen.summary()
    plot_model(gen, to_file="generator.png", show_shapes=True)

    disc = PatchGAN((256, 256, 3))
    disc.summary()
    plot_model(disc, to_file="discriminator.png", show_shapes=True)
