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
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50V2

# parar mensagens do lado Python
tf.get_logger().setLevel('ERROR')

# ============================================================
# FICHEIROS (GRID SEARCH - TRAIN_700)
# ============================================================
SAVE_JSON_BEST = "grid_best_resnet_train700.json"
SAVE_CSV_ALL   = "grid_results_resnet_train700.csv"
FINAL_MODEL_PATH = "final_resnet_grid_train700.keras"
FINAL_HISTORY_PATH = "history_resnet_grid_train700.pkl"

JSON_LOCK = threading.Lock()

# ============================================================
# DATASETS  (treino na pasta train_700, validação na mesma pasta de sempre)
# ============================================================
def ds(path, training):
    d = keras.preprocessing.image_dataset_from_directory(
        path, image_size=(224, 224), batch_size=32,
        label_mode="categorical", shuffle=training
    )
    return d.map(lambda x, y: (tf.cast(x, tf.float32)/255.0, y)).prefetch(tf.data.AUTOTUNE)

print("\n[GRID] A carregar datasets de treino e validação...")
train_ds = ds("../../dataset/train_700", True)
val_ds   = ds("../../dataset/validation", False)
print("[GRID] Datasets carregados.\n")

# lista para guardar TODOS os resultados (para CSV)
resultados_grid = []

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
# FUNÇÃO DE AVALIAÇÃO PARA UMA COMBINAÇÃO DO GRID
# ============================================================
def avaliar_combinacao(p, raw_tag=None):
    """
    Avalia um conjunto de hiperparâmetros p (dict com units, lr, dropout).
    Guarda info na lista global e atualiza JSON se for o melhor.
    """
    global resultados_grid

    print(f"\n[GRID][TESTE] units={p['units']} | lr={p['lr']:.6f} | dropout={p['dropout']:.3f}")

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

        print(f"[GRID][RESULTADO] val_acc={acc_val:.4f} | val_loss={loss_val:.4f} | fitness={fitness:.4f}")

        # guardar resultado
        resultados_grid.append({
            "units": p["units"],
            "lr": p["lr"],
            "dropout": p["dropout"],
            "val_accuracy": acc_val,
            "val_loss": loss_val,
            "fitness": fitness,
            "epochs_treinadas": epochs_treinadas
        })

        # JSON de melhor resultado (mesmo template que GWO)
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
                    print(f"[GRID][AVISO] Não foi possível ler {SAVE_JSON_BEST}: {e}")

            if acc_val > score_antigo:
                print(f"[GRID][CHECKPOINT] Novo melhor! Acc={acc_val:.4f} (antigo={score_antigo:.4f}). A salvar JSON...")
                try:
                    with open(SAVE_JSON_BEST, "w") as f:
                        json.dump(novo_resultado, f, indent=4)
                except Exception as e:
                    print(f"[GRID][ERRO] Não foi possível salvar {SAVE_JSON_BEST}: {e}")
            else:
                print(f"[GRID][CHECKPOINT] Resultado (Acc: {acc_val:.4f}) não superou o melhor (Acc={score_antigo:.4f}).")

        keras.backend.clear_session()
        del model

    except Exception as e:
        print(f"[GRID][ERRO NO TREINO] {e}")
        keras.backend.clear_session()

# ============================================================
# DEFINIÇÃO DO GRID
# ============================================================
UNITS_OPTIONS    = [128, 256, 512, 1024]
DROPOUT_OPTIONS  = [0.30, 0.40, 0.50, 0.60, 0.70]
LR_OPTIONS       = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2] # 4 × 5 × 7 = 140 combinações máx.

total_combos = len(UNITS_OPTIONS) * len(DROPOUT_OPTIONS) * len(LR_OPTIONS)

print("--- OTIMIZAÇÃO DE HIPERPARÂMETROS COM GRID SEARCH (PASSO 1/2) ---")
print(f"Total de combinações a testar: {total_combos}\n")

start_grid = time.time()

comb_count = 0
for units in UNITS_OPTIONS:
    for dropout in DROPOUT_OPTIONS:
        for lr in LR_OPTIONS:
            comb_count += 1
            print(f"\n[GRID] Combinação {comb_count}/{total_combos}")
            p = {
                "units": int(units),
                "lr": float(lr),
                "dropout": float(dropout)
            }
            # raw_tag só para manter a mesma estrutura do JSON
            raw_tag = [units, lr, dropout]
            avaliar_combinacao(p, raw_tag=raw_tag)

end_grid = time.time()

print("\n" + "=" * 50)
print(f"[GRID] OTIMIZAÇÃO COMPLETA em {(end_grid - start_grid) / 3600:.2f} horas")
print("=" * 50)

# ============================================================
# GUARDAR TODOS OS RESULTADOS EM CSV
# ============================================================
if len(resultados_grid) > 0:
    campos = ["units", "lr", "dropout", "val_accuracy", "val_loss", "fitness", "epochs_treinadas"]
    try:
        with open(SAVE_CSV_ALL, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=campos)
            writer.writeheader()
            writer.writerows(resultados_grid)
        print(f"\n[GRID][FICHEIRO] Resultados completos guardados em: {SAVE_CSV_ALL}")
    except Exception as e:
        print(f"[GRID][ERRO] Não foi possível salvar CSV {SAVE_CSV_ALL}: {e}")
else:
    print("\n[GRID][AVISO] Nenhum resultado para guardar em CSV.")

# ============================================================
# LER MELHOR RESULTADO DO JSON E RE-TREINAR MODELO FINAL
# ============================================================
print("\n[GRID] A ler melhor resultado do JSON para re-treino final...\n")

try:
    with open(SAVE_JSON_BEST, "r") as f:
        dados_finais = json.load(f)
    best_params = dados_finais["parametros_descodificados"]
    print(f"  Melhor Accuracy (Validação): {dados_finais['melhor_accuracy']:.6f}")
    print("\n  Melhor conjunto de Hiperparâmetros (GRID):")
    for k, v in best_params.items():
        print(f"    {k}: {v:.6f}" if isinstance(v, float) else f"    {k}: {v}")
except Exception as e:
    print(f"[GRID][ERRO] Não foi possível ler {SAVE_JSON_BEST}: {e}")
    best_params = None

print("=" * 50)

if best_params:
    print("\n--- [GRID] RE-TREINO FINAL COM MELHORES HIPERPARÂMETROS (PASSO 2/2) ---\n")

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

    print("[GRID] A iniciar treino final (até 50 épocas, patience=10)...")

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
    print("[GRID] RE-TREINO FINAL CONCLUÍDO.")
    print(f"Modelo final guardado em: {FINAL_MODEL_PATH}")
    print(f"Histórico guardado em: {FINAL_HISTORY_PATH}")
    print("=" * 50)
else:
    print("\n[GRID][AVISO] Não foi possível efetuar o re-treino final porque não há parâmetros válidos.")
