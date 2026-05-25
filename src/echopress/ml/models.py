from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_name: str = "fft_mlp"
    hidden_dims: list[int] = Field(default_factory=lambda: [256, 128, 64])
    dropout: float = 0.2
    learning_rate: float = 1e-3


def build_model(input_dim: int, config: ModelConfig):
    try:
        import tensorflow as tf
    except Exception as exc:  # pragma: no cover - dependency/runtime guard
        raise RuntimeError("TensorFlow is required for model building") from exc

    if config.model_name != "fft_mlp":
        raise ValueError(f"Unsupported model: {config.model_name}")

    inputs = tf.keras.Input(shape=(input_dim,), name="fft_features")
    x = inputs
    for i, width in enumerate(config.hidden_dims):
        x = tf.keras.layers.Dense(width, activation="relu", name=f"dense_{i}")(x)
        if config.dropout > 0:
            x = tf.keras.layers.Dropout(config.dropout, name=f"dropout_{i}")(x)
    outputs = tf.keras.layers.Dense(1, activation="linear", name="pressure")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=config.model_name)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss="mse",
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model
