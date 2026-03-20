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
from SwarmPackagePy import gwo

# parar mensagens do lado Python
tf.get_logger().setLevel('ERROR')

# ============================================================
# FICHEIROS (ajustar o nome consoante a sub-pasta)
# ============================================================
SAVE_JSON_BEST = "gwo_best_resnet_train700.json"
SAVE_CSV_ALL   = "gwo_results_resnet_train700.csv"
FINAL_MODEL_PATH = "final_resnet_gwo_train700.keras"
FINAL_HISTORY_PATH = "history_resnet_gwo_train700.pkl"

JSON_LOCK = threading.Lock()

# ============================================================
# DATASETS  (AQUI MUDAR APENAS A PASTA DE TREINO: train_150, 300, 500, 700)
# ============================================================
def ds(path, training):
    d = keras.preprocessing.image_dataset_from_directory( # d é uma lista que guarda 2valores (imagem=x, etiqueta=y) | o x é uma matriz multidimensional (224, 224, 3), onde o '224, 224' é altura e largura de cada imagem e o 3 é o num de cores (RGB), dentro dessa matriz existem os valores da cor de cada pixel (p ex: [255, 0, 0] -> Vermelho) | já o y é a classe de cada imagem, como nós estamos a usar 'label_mode'="categorical", em vez de o y guardar o nome da classe, guarda num vetor One-Hot, onde basicamente tem o numero de colunas = ao numero de classes que existe (p ex: [1,0,0,0,0])), e a classe correta da imagem aparece com o número '1', no exemplo anterior a imagem seria da classe "convertible", sendo a 1a classe das 5 por ordem alfabética
        path, image_size=(224, 224), batch_size=32,
        label_mode="categorical", shuffle=training
    )
    return d.map(lambda x, y: (tf.cast(x, tf.float32)/255.0, y)).prefetch(tf.data.AUTOTUNE) # passa os valores das cores dos pixels (daquela matriz(x) em d para float e divide por 255 (valor max do RGB) para normalizar os dados (o intervalo fica entre 0.0 e 1.0) o que faz com que as redes neuronais treinem melhor | O comando .prefetch(tf.data.AUTOTUNE) no train.py garante que o próximo lote de dados está sempre pronto na GPU/CPU enquanto o lote atual está a ser processado, maximizando o desempenho do treino

print("\nA carregar datasets de treino e validação...")
train_ds = ds("../../dataset/train_700", True)          
val_ds   = ds("../../dataset/validation", False)
print("Datasets carregados.\n")

# lista para guardar TODOS os resultados (para CSV)
resultados_gwo = []

# ============================================================
# DECODIFICAÇÃO DOS HIPERPARÂMETROS (3 dimensões)
# ============================================================
def decode_params(params_0_1):
    """
    params_0_1: vetor [u0, u1, u2] no intervalo [0,1]
        u0 -> nº neurónios Dense (128, 256, 512)
        u1 -> learning rate [1e-5, 1e-3] em escala log
        u2 -> dropout [0.30, 0.70]
    """

    u0, u1, u2 = params_0_1

    # nº de neurónios (categórico)
    if u0 < 1/3:
        units = 128
    elif u0 < 2/3:
        units = 256
    else:
        units = 512

    # learning rate (escala log)
    lr_min, lr_max = 1e-5, 1e-3
    lr = 10 ** (np.log10(lr_min) + u1 * (np.log10(lr_max) - np.log10(lr_min)))

    # dropout
    d_min, d_max = 0.30, 0.70
    dropout = d_min + u2 * (d_max - d_min)

    return {
        "units": int(units),
        "lr": float(lr),
        "dropout": float(dropout)
    }

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

    # rede densa
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
# FUNÇÃO DE AVALIAÇÃO PARA O GWO
# ============================================================
def avaliar_modelo(params_0_1):
    """
    Função de fitness para o GWO.
    Recebe params_0_1 (3D, valores entre 0 e 1) e devolve 1 - accuracy_val.
    """
    global resultados_gwo

    try:
        p = decode_params(params_0_1)
    except Exception as e:
        print(f"[ERRO] ao descodificar parâmetros: {e}")
        return 1.0  # pior fitness

    print(f"\n[TESTE] units={p['units']} | lr={p['lr']:.6f} | dropout={p['dropout']:.3f}")

    try:
        model = build_model_from_params(p)

        earlyStopping = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=7, # 7
            restore_best_weights=True
        )

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=30, # 30
            callbacks=[earlyStopping],
            verbose=0
        )

        # obter nº de épocas realmente treinadas
        epochs_treinadas = len(history.history["loss"])

        loss_val, acc_val = model.evaluate(val_ds, verbose=0)
        fitness = 1.0 - acc_val

        print(f"[RESULTADO] Accuracy={acc_val:.4f} | Loss={loss_val:.4f} | Fitness={fitness:.4f}")

        # guardar resultados na lista (para CSV no final)
        resultados_gwo.append({
            "units": p["units"],
            "lr": p["lr"],
            "dropout": p["dropout"],
            "val_accuracy": acc_val,
            "val_loss": loss_val,
            "fitness": fitness,
            "epochs_treinadas": epochs_treinadas
        })

        # UPDATE DO MELHOR RESULTADO NO JSON
        novo_resultado = {
            "melhor_accuracy": acc_val,
            "parametros_descodificados": p,
            "parametros_raw_0_1": list(params_0_1)
        }

        with JSON_LOCK:
            score_antigo = -1.0
            if os.path.exists(SAVE_JSON_BEST):
                try:
                    with open(SAVE_JSON_BEST, "r") as f:
                        dados_antigos = json.load(f)
                        score_antigo = dados_antigos.get("melhor_accuracy", -1.0)
                except Exception as e:
                    print(f"[AVISO] Não foi possível ler {SAVE_JSON_BEST}: {e}")

            if acc_val > score_antigo:
                print(f"[CHECKPOINT] Novo melhor resultado! Acc: {acc_val:.4f} (anterior: {score_antigo:.4f}). A salvar JSON...")
                try:
                    with open(SAVE_JSON_BEST, "w") as f:
                        json.dump(novo_resultado, f, indent=4)
                except Exception as e:
                    print(f"[ERRO] Não foi possível salvar {SAVE_JSON_BEST}: {e}")
            else:
                print(f"[CHECKPOINT] Resultado (Acc: {acc_val:.4f}) não superou o melhor (Acc: {score_antigo:.4f}).")

        # limpar sessão
        keras.backend.clear_session()
        del model

        return fitness

    except Exception as e:
        print(f"[ERRO NO TREINO] {e}")
        keras.backend.clear_session()
        return 1.0

# ============================================================
# PARÂMETROS DO GWO
# ============================================================
dim = 3 # numero de hiperparametros a otimizar pelo otimizador [units, lr, dropout]
lb = np.array([0.0] * dim) # array com os valores de  lowbound  para cada hiperparametro ([0.0, 0.0, 0.0]) 
ub = np.array([1.0] * dim) # array com os valores de upperbound para cada hiperparametro ([1.0, 1.0, 1.0]) 

pop_size = 10      # 10 agentes
iterations = 10    # 10 iterações

print("--- OTIMIZAÇÃO DE HIPERPARÂMETROS COM GWO (PASSO 1/2) ---")
print(f"Dimensão: {dim}, População: {pop_size}, Iterações: {iterations}")
print(f"Total de modelos 'proxy' a treinar: {pop_size * iterations}\n")

print(f"A iniciar GWO...")
start_gwo = time.time()
gwo(pop_size, avaliar_modelo, lb, ub, dim, iterations)
end_gwo = time.time()

print("\n" + "=" * 50)
print(f"OTIMIZAÇÃO GWO COMPLETA em {(end_gwo - start_gwo) / 3600:.2f} horas")
print("=" * 50)

# ============================================================
# GUARDAR TODOS OS RESULTADOS EM CSV (para Excel)
# ============================================================
if len(resultados_gwo) > 0:
    campos = ["units", "lr", "dropout", "val_accuracy", "val_loss", "fitness", "epochs_treinadas"]
    try:
        with open(SAVE_CSV_ALL, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=campos)
            writer.writeheader()
            writer.writerows(resultados_gwo)
        print(f"\n[FICHEIRO] Resultados completos da otimização guardados em: {SAVE_CSV_ALL}")
    except Exception as e:
        print(f"[ERRO] Não foi possível salvar CSV {SAVE_CSV_ALL}: {e}")
else:
    print("\n[AVISO] Nenhum resultado para guardar em CSV.")

# ============================================================
# LER MELHOR RESULTADO DO JSON E RE-TREINAR MODELO FINAL
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

if best_params:
    print("\n--- RE-TREINO FINAL COM MELHORES HIPERPARÂMETROS (PASSO 2/2) ---\n")

    final_model = build_model_from_params(best_params)

    final_callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10, # 10
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
        epochs=50, # 50
        callbacks=final_callbacks,
        verbose=1
    )

    with open(FINAL_HISTORY_PATH, "wb") as f:
        pickle.dump(history.history, f)

    print("\n" + "=" * 50)
    print("RE-TREINO FINAL CONCLUÍDO.")
    print(f"Modelo final guardado em: {FINAL_MODEL_PATH}")
    print(f"Histórico guardado em: {FINAL_HISTORY_PATH}")
    print("=" * 50)
else:
    print("\n[AVISO] Não foi possível efetuar o re-treino final porque não há parâmetros válidos.")
