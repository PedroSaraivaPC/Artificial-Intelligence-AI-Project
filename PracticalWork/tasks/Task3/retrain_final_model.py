import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import json
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50V2

tf.get_logger().setLevel('ERROR')

# ============================================================
# FICHEIROS (AJUSTAR NOMES SE NECESSÁRIO)
# ============================================================
SAVE_JSON_BEST     = "gwo_best_resnet_train700.json"   # JSON vindo da otimização
FINAL_MODEL_PATH    = "final_model.keras"
FINAL_HISTORY_PATH  = "history.pkl"

# ============================================================
# DATASETS  (train_700 e validation, como no teu script original)
# ============================================================
def ds(path, training):
    d = keras.preprocessing.image_dataset_from_directory(
        path,
        image_size=(224, 224),
        batch_size=32,
        label_mode="categorical",
        shuffle=training
    )
    return d.map(lambda x, y: (tf.cast(x, tf.float32)/255.0, y)).prefetch(tf.data.AUTOTUNE)

print("\nA carregar datasets de treino e validação...")
train_ds = ds("../../dataset/train_700", True)
val_ds   = ds("../../dataset/validation", False)
print("Datasets carregados.\n")

# ============================================================
# CONSTRUIR MODELO ResNet50V2 + TOP LAYER A PARTIR DE p
# (EXACTAMENTE COMO NO TEU SCRIPT)
# ============================================================
def build_model_from_params(p):
    base_model = ResNet50V2(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(p["units"], activation="relu")(x)
    x = layers.Dropout(p["dropout"])(x)

    outputs = layers.Dense(5, activation="softmax")(x)

    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=p["lr"]),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

# ============================================================
# LER MELHOR RESULTADO DO JSON (DO GWO)
# ============================================================
print("\nA ler melhor resultado do JSON para re-treino final...\n")

try:
    with open(SAVE_JSON_BEST, "r") as f:
        dados_finais = json.load(f)

    best_params = dados_finais["parametros_descodificados"]

    print(f"  Melhor Accuracy (Validação): {dados_finais['melhor_accuracy']:.6f}")
    print("\n  Melhor conjunto de Hiperparâmetros:")
    for k, v in best_params.items():
        print(f"    {k}: {v:.6f}" if isinstance(v, float) else f"    {k}: {v}")

except Exception as e:
    print(f"[ERRO] Não foi possível ler {SAVE_JSON_BEST}: {e}")
    best_params = None

print("=" * 50)

# ============================================================
# RE-TREINO FINAL COM MELHORES PARÂMETROS
# ============================================================
if best_params:
    print("\n--- RE-TREINO FINAL COM MELHORES HIPERPARÂMETROS (PASSO ÚNICO) ---\n")

    final_model = build_model_from_params(best_params)

    final_callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=FINAL_MODEL_PATH,
            monitor="val_loss",
            save_best_only=True
        )
    ]

    print("A iniciar treino final (até 50 épocas, patience=10)...")

    history = final_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=50,
        callbacks=final_callbacks,
        verbose=1
    )

    # guardar histórico
    with open(FINAL_HISTORY_PATH, "wb") as f:
        pickle.dump(history.history, f)

    print("\n" + "=" * 50)
    print("RE-TREINO FINAL CONCLUÍDO.")
    print(f"Modelo final guardado em: {FINAL_MODEL_PATH}")
    print(f"Histórico guardado em: {FINAL_HISTORY_PATH}")
    print("=" * 50)

else:
    print("\n[AVISO] Não foi possível efetuar o re-treino final porque não há parâmetros válidos.")
