from __future__ import annotations

def build_model(n_features, cfg):
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required. Install with: pip install -r requirements-ml.txt") from exc
    mcfg = cfg.get("model", {})
    units = mcfg.get("hidden_units", [256,128,64]); dropout=float(mcfg.get("dropout",0.1)); batch_norm=bool(mcfg.get("batch_norm",True))
    inp = tf.keras.Input(shape=(n_features,))
    x = inp
    for i,u in enumerate(units):
        x = tf.keras.layers.Dense(int(u))(x)
        if batch_norm and i < 2: x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        if i < 2 and dropout>0: x = tf.keras.layers.Dropout(dropout)(x)
    out = tf.keras.layers.Dense(1, activation=mcfg.get("output_activation","linear"))(x)
    model = tf.keras.Model(inp, out)
    return model
