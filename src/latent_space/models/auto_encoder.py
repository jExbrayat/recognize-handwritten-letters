import os

os.environ["KERAS_BACKEND"] = "torch"

import numpy as np

from keras.src.losses import MeanSquaredError
from keras.src.optimizers import Adam

from keras import Model

from keras.src.layers import (
    Dense,
    Input,
    MaxPooling2D,
    Conv2DTranspose,
    Conv2D,
    Reshape,
    Flatten,
)

from matplotlib import pyplot as plt

from src.classification.training.callbacks import history, early_stopping


def build_encoder(latent_dimension: int = 26) -> Model:
    # Input
    input_layer = Input(shape=(784,))
    reshape_layer = Reshape((28, 28, 1))(input_layer)

    # Encoder
    conv_layer1 = Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same")(
        reshape_layer
    )  # 28x28x32
    maxpool_layer1 = MaxPooling2D(padding="same")(conv_layer1)  # 14x14x32

    conv_layer2 = Conv2D(16, kernel_size=(3, 3), activation="relu", padding="same")(
        maxpool_layer1
    )  # 28x28x32
    maxpool_layer2 = MaxPooling2D(padding="same")(conv_layer2)  # 14x14x32

    flatten_layer = Flatten()(maxpool_layer2)
    dense_layer = Dense(latent_dimension)(flatten_layer)

    # Model
    encoder_model = Model(inputs=input_layer, outputs=dense_layer, name="encoder")

    return encoder_model


def build_decoder(latent_dimension: int = 26) -> Model:
    # Input
    input_layer = Input(shape=(latent_dimension,))

    # Decoder
    # Projection vers un tenseur de forme (7, 7, 32)
    x = Dense(7 * 7 * 32, activation="relu")(input_layer)
    x = Reshape((7, 7, 32))(x)

    # Upsampling + convolution transpose pour reconstruire
    x = Conv2DTranspose(
        16, kernel_size=3, strides=2, padding="same", activation="relu"
    )(x)  # -> (14, 14, 16)
    x = Conv2DTranspose(
        32, kernel_size=3, strides=2, padding="same", activation="relu"
    )(x)  # -> (28, 28, 32)

    # Dernière couche pour revenir à 1 canal (grayscale)
    output_img = Conv2DTranspose(
        1, kernel_size=3, padding="same", activation="sigmoid"
    )(x)  # -> (28, 28, 1)

    # Output
    # Flatten pour correspondre à l'entrée (784,)
    output_layer = Reshape((784,))(output_img)

    # Model
    decoder_model = Model(inputs=input_layer, outputs=output_layer, name="decoder")

    return decoder_model


def build_auto_encoder() -> tuple[Model, Model, Model]:
    encoder = build_encoder()
    decoder = build_decoder()

    input_layer = encoder.input
    encoded = encoder(input_layer)
    decoded = decoder(encoded)

    auto_encoder = Model(input_layer, decoded)
    auto_encoder.compile(loss=MeanSquaredError(), optimizer=Adam())

    return auto_encoder, encoder, decoder


class AutoEncoder:
    def __init__(self):
        self.autoencoder, self.encoder, self.decoder = build_auto_encoder()
        self.history = history()

    def encode(self, x: np.ndarray) -> np.ndarray:
        return self.encoder.predict(x)

    def decode(self, x: np.ndarray) -> np.ndarray:
        return self.decoder.predict(x)

    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        return self.autoencoder.predict(x)

    def fit(self, x_train: np.ndarray, epochs: int = 10) -> None:
        es = early_stopping()

        self.autoencoder.fit(
            x_train,
            x_train,
            epochs=epochs,
            validation_split=0.2,
            callbacks=[es, self.history],
        )

    def plot_training_curve(self) -> plt.Figure:
        fig = plt.figure()

        plt.plot(self.history.history["loss"], label="MSE loss")
        plt.plot(self.history.history["val_loss"], label="MSE val_loss")

        plt.title("Training curve")
        plt.xlabel("epoch")
        plt.ylabel("loss")

        plt.legend()

        return fig

    def save(self) -> None:
        raise NotImplementedError

    def visualize(self, x: np.ndarray, n: int = 5) -> plt.Figure:
        decoded_imgs = self.reconstruct(x[:n])
        diff_imgs = x[:n] - decoded_imgs

        fig = plt.figure(figsize=(8, 4 * n))

        for i in range(n):
            # Original
            plt.subplot(n, 3, i * 3 + 1)
            plt.imshow(x[i].reshape(28, 28), cmap="gray")
            plt.title("Original")
            plt.axis("off")

            # Reconstruit
            plt.subplot(n, 3, i * 3 + 2)
            plt.imshow(decoded_imgs[i].reshape(28, 28), cmap="gray")
            plt.title("Reconstruit")
            plt.axis("off")

            # Différence
            plt.subplot(n, 3, i * 3 + 3)
            plt.imshow(
                diff_imgs[i].reshape(28, 28), cmap="seismic", vmin=-1.0, vmax=1.0
            )
            plt.title("Différence")
            plt.axis("off")

        return fig
