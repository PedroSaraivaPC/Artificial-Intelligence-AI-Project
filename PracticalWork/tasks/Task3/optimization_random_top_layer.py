import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # tirar mensagens C++ do TF

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import time
import json
import threading
import csv
import pickle
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50V2

# parar mensagens do lado Python
tf.get_logger().setLevel('ERROR')

# ============================================================
# FICHEIROS (RANDOM SEARCH - TRAIN_700)
# ============================================================
SAVE_JSON_BEST = "random_best_resnet_train700.json"
SAVE_CSV_ALL   = "random_results_resnet_train700.csv"
FINAL_MODEL_PATH = "final_resnet_random_train700.keras"
FINAL_HISTORY_PATH = "history_resnet_random_train700.pkl"

JSON_LOCK = threading.Lock()

# ============================================================
# DATASETS
# ============================================================
def ds(path, training):
    d = keras.preprocessing.image_dataset_from_directory(
        path, image_size=(224, 224), batch_size=32,
        label_mode="categorical", shuffle=training
    )
    return d.map(lambda x, y: (tf.cast(x, tf.float32)/255.0, y)).prefetch(tf.data.AUTOTUNE)

print("\n[RANDOM] A carregar datasets de treino e validação...")
train_ds = ds("../../dataset/train_700", True)
val_ds   = ds("../../dataset/validation", False)
print("[RANDOM] Datasets carregados.\n")

# lista para guardar TODOS os resultados (para CSV)
resultados_random = []

# ============================================================
# CONSTRUIR MODELO ResNet50V2 + TOP LAYER A PARTIR DE p
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
# FUNÇÃO DE AVALIAÇÃO PARA UMA COMBINAÇÃO RANDOM
# ============================================================
def avaliar_random(p, raw_tag=None):
    """
    Avalia um conjunto de hiperparâmetros p (dict com units, lr, dropout).
    Guarda info na lista global e atualiza JSON se for o melhor.
    """
    global resultados_random

    print(f"\n[RANDOM][TESTE] units={p['units']} | lr={p['lr']:.6f} | dropout={p['dropout']:.3f}")

    try:
        model = build_model_from_params(p)

        earlyStopping = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=7,
            restore_best_weights=True
        )

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=30,
            callbacks=[earlyStopping],
            verbose=0
        )

        epochs_treinadas = len(history.history["loss"])
        loss_val, acc_val = model.evaluate(val_ds, verbose=0)
        fitness = 1.0 - acc_val

        print(f"[RANDOM][RESULTADO] val_acc={acc_val:.4f} | val_loss={loss_val:.4f} | fitness={fitness:.4f}")

        # guardar resultado
        resultados_random.append({
            "units": p["units"],
            "lr": p["lr"],
            "dropout": p["dropout"],
            "val_accuracy": acc_val,
            "val_loss": loss_val,
            "fitness": fitness,
            "epochs_treinadas": epochs_treinadas
        })

        # JSON de melhor resultado (mesmo formato do GWO)
        novo_resultado = {
            "melhor_accuracy": acc_val,
            "parametros_descodificados": p,
            "parametros_raw_0_1": raw_tag if raw_tag is not None else [None, None, None]
        }

        with JSON_LOCK:
            score_antigo = -1.0
            if os.path.exists(SAVE_JSON_BEST):
                try:
                    with open(SAVE_JSON_BEST, "r") as f:
                        dados_antigos = json.load(f)
                        score_antigo = dados_antigos.get("melhor_accuracy", -1.0)
                except Exception as e:
                    print(f"[RANDOM][AVISO] Não foi possível ler {SAVE_JSON_BEST}: {e}")

            if acc_val > score_antigo:
                print(f"[RANDOM][CHECKPOINT] Novo melhor! Acc={acc_val:.4f} (antigo={score_antigo:.4f}). A salvar JSON...")
                try:
                    with open(SAVE_JSON_BEST, "w") as f:
                        json.dump(novo_resultado, f, indent=4)
                except Exception as e:
                    print(f"[RANDOM][ERRO] Não foi possível salvar {SAVE_JSON_BEST}: {e}")
            else:
                print(f"[RANDOM][CHECKPOINT] Resultado (Acc: {acc_val:.4f}) não superou o melhor (Acc={score_antigo:.4f}).")

        keras.backend.clear_session()
        del model

    except Exception as e:
        print(f"[RANDOM][ERRO NO TREINO] {e}")
        keras.backend.clear_session()

# ============================================================
# RANDOM SEARCH CONFIG
# ============================================================
UNITS_OPTIONS = [128, 256, 512]
DROPOUT_MIN, DROPOUT_MAX = 0.30, 0.70
LR_MIN, LR_MAX = 1e-5, 1e-3

NUM_RANDOM_TRIALS = 140  # podes ajustar

print("--- OTIMIZAÇÃO DE HIPERPARÂMETROS COM RANDOM SEARCH (PASSO 1/2) ---")
print(f"Número de amostras aleatórias a testar: {NUM_RANDOM_TRIALS}\n")

start_random = time.time()

for trial in range(1, NUM_RANDOM_TRIALS + 1):
    print(f"\n[RANDOM] Trial {trial}/{NUM_RANDOM_TRIALS}")

    # units categórico
    units = random.choice(UNITS_OPTIONS)

    # dropout contínuo uniforme
    dropout = random.uniform(DROPOUT_MIN, DROPOUT_MAX)

    # lr em escala log-uniforme
    u = random.random()  # [0,1]
    lr = 10 ** (np.log10(LR_MIN) + u * (np.log10(LR_MAX) - np.log10(LR_MIN)))

    p = {
        "units": int(units),
        "lr": float(lr),
        "dropout": float(dropout)
    }

    raw_tag = [units, lr, dropout]  # só para manter o mesmo campo no JSON
    avaliar_random(p, raw_tag=raw_tag)

end_random = time.time()

print("\n" + "=" * 50)
print(f"[RANDOM] OTIMIZAÇÃO COMPLETA em {(end_random - start_random) / 3600:.2f} horas")
print("=" * 50)

# ============================================================
# GUARDAR TODOS OS RESULTADOS EM CSV
# ============================================================
if len(resultados_random) > 0:
    campos = ["units", "lr", "dropout", "val_accuracy", "val_loss", "fitness", "epochs_treinadas"]
    try:
        with open(SAVE_CSV_ALL, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=campos)
            writer.writeheader()
            writer.writerows(resultados_random)
        print(f"\n[RANDOM][FICHEIRO] Resultados completos guardados em: {SAVE_CSV_ALL}")
    except Exception as e:
        print(f"[RANDOM][ERRO] Não foi possível salvar CSV {SAVE_CSV_ALL}: {e}")
else:
    print("\n[RANDOM][AVISO] Nenhum resultado para guardar em CSV.")

# ============================================================
# LER MELHOR RESULTADO DO JSON E RE-TREINAR MODELO FINAL
# ============================================================
print("\n[RANDOM] A ler melhor resultado do JSON para re-treino final...\n")

try:
    with open(SAVE_JSON_BEST, "r") as f:
        dados_finais = json.load(f)
    best_params = dados_finais["parametros_descodificados"]
    print(f"  Melhor Accuracy (Validação): {dados_finais['melhor_accuracy']:.6f}")
    print("\n  Melhor conjunto de Hiperparâmetros (RANDOM):")
    for k, v in best_params.items():
        print(f"    {k}: {v:.6f}" if isinstance(v, float) else f"    {k}: {v}")
except Exception as e:
    print(f"[RANDOM][ERRO] Não foi possível ler {SAVE_JSON_BEST}: {e}")
    best_params = None

print("=" * 50)

if best_params:
    print("\n--- [RANDOM] RE-TREINO FINAL COM MELHORES HIPERPARÂMETROS (PASSO 2/2) ---\n")

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

    print("[RANDOM] A iniciar treino final (até 50 épocas, patience=10)...")

    history = final_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=50,
        callbacks=final_callbacks,
        verbose=1
    )

    with open(FINAL_HISTORY_PATH, "wb") as f:
        pickle.dump(history.history, f)

    print("\n" + "=" * 50)
    print("[RANDOM] RE-TREINO FINAL CONCLUÍDO.")
    print(f"Modelo final guardado em: {FINAL_MODEL_PATH}")
    print(f"Histórico guardado em: {FINAL_HISTORY_PATH}")
    print("=" * 50)
else:
    print("\n[RANDOM][AVISO] Não foi possível efetuar o re-treino final porque não há parâmetros válidos.")
