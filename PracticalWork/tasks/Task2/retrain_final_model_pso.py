import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # parar tirar mensagens do back-end em C++ do tensorflow
import tensorflow as tf, pickle

import numpy as np
import json       
import warnings
from tensorflow import keras
from tensorflow.keras import layers

# parar tirar mensagens do lado do python do tensorflow
tf.get_logger().setLevel('ERROR')
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

JSON_RESULT_FILE = "pso_best_result.json" # ficheiro para ler os hiperparametros
FINAL_EPOCHS = 100
FINAL_PATIENCE = 10
MODEL_SAVE_PATH = "final_model_pso.keras" # ficheiro para salvar o melhor modelo final treinado
HISTORY_SAVE_PATH = "history_pso.pkl" # ficheiro para salvar o historico do treino para fazer plot dos graficos de loss e accuracy

# pre-processamento e normalizacao das imagens
def ds(path, training):
    d = keras.preprocessing.image_dataset_from_directory(
        path, image_size=(128, 128), batch_size=32,
        label_mode="categorical", shuffle=training
    )
    return d.map(lambda x,y: (tf.cast(x, tf.float32)/255.0, y)).prefetch(tf.data.AUTOTUNE)

print("A carregar datasets de treino e validação...")
train_ds = ds("../../dataset/train", True)
val_ds = ds("../../dataset/validation", False)
print("Datasets carregados.")

# cria a arquitetura da CNN já com os parametros gerados pelo algoritmo
def build_model_from_params(p):
    
    k = (p['kernel'], p['kernel']) # ex: (3,3) ou (5,5)
    
    return keras.Sequential([
        # Input (128x128 pixeis RGB(3 canais de cores))
        layers.Input((128,128,3)),
        
        # Bloco 1
        layers.Conv2D(p['f1'], k, padding='same', use_bias=False),
        layers.BatchNormalization(), layers.Activation('relu'),
        layers.Conv2D(p['f1'], k, padding='same', use_bias=False),
        layers.BatchNormalization(), layers.Activation('relu'),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(p['d1']),
        
        # Bloco 2
        layers.Conv2D(p['f2'], k, padding='same', use_bias=False), 
        layers.BatchNormalization(), layers.Activation('relu'),
        layers.Conv2D(p['f2'], k, padding='same', use_bias=False), 
        layers.BatchNormalization(), layers.Activation('relu'),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(p['d2_3']),
        
        # Bloco 3
        layers.Conv2D(p['f3_dense'], k, padding='same', use_bias=False), 
        layers.BatchNormalization(), layers.Activation('relu'),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(p['d2_3']),
        layers.Conv2D(p['f3_dense'], k, padding='same', use_bias=False), 
        layers.BatchNormalization(), layers.Activation('relu'),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(p['d2_3']),
        
        # Classificador
        layers.GlobalAveragePooling2D(),
        layers.Dense(p['f3_dense'], use_bias=False),
        layers.BatchNormalization(), layers.Activation('relu'),
        layers.Dropout(p['d_dense']),
        layers.Dense(5, activation='softmax')
    ])

# re-treina o modelo final com os melhores valores dos hiperparametros encontrados pelo otimizador
print("\n" + "=" * 50)
print("--- RE-TREINO FINAL (A PARTIR DO JSON) ---")
print(f"A ler os melhores parâmetros de {JSON_RESULT_FILE}...")

# le o melhor resultado do JSON para reetreinar o modelo com os melhores parametros encontrados na otimizacao
best_params_pso = None
try:
    with open(JSON_RESULT_FILE, 'r') as f:
        dados_finais = json.load(f)
    best_params_pso = dados_finais["parametros_descodificados"]   
    print(f"  Melhor Accuracy (Validação) registada: {dados_finais.get('melhor_accuracy', 'N/A'):.6f}")
    print("\n  Melhor conjunto de Hiperparâmetros (lido do JSON):")
    for key, value in best_params_pso.items():
        print(f"    {key}: {value:.6f}" if isinstance(value, float) else f"    {key}: {value}")
except Exception as e:
    print(f"[ERRO] Não foi possível ler o ficheiro de resultados {JSON_RESULT_FILE}: {e}")
    print("Re-treino final cancelado.")
print("=" * 50)

# re-treina o modelo final com os melhores valores dos hiperparametros encontrados pelo otimizador. Só executa o treino se os parâmetros foram carregados com sucesso
if best_params_pso:
    print("\n--- A INICIAR RE-TREINO FINAL ---")
    print("A preparar o modelo final com os melhores parâmetros...")

    # callbacks para o treino final
    final_callbacks = [
        keras.callbacks.EarlyStopping( # callback que serve APENAS PARA parar o treino se não melhorar
            monitor="val_loss", # controla pela loss
            patience=FINAL_PATIENCE, # nº de épocas sem melhorar até parar
            restore_best_weights=True # no final de cada epoca salva na RAM o melhor modelo se for o com melhor accuracy para no final do treino fazer restauro do modelo e salvar(model.save), mas neste caso nao se está a fazer save dessa forma, mas sim com o callback 'ModelCheckpoint' para ser mais seguro, que salva mesmo já o ficheiro .keras ao final da epoca (caso seja o com melhor valor de acc) antes que dê problema com a RAM durante o treino
        ),
        keras.callbacks.ModelCheckpoint( # callback que serve para salvar o melhor modelo no ficheiro .keras a cada epoca (se for melhor que o anterior melhor)
            filepath=MODEL_SAVE_PATH, # nome do ficheiro
            monitor="val_loss", # controla pela loss
            save_best_only=True # no final de cada epoca salva o ficheiro .keras com o modelo se for o com melhor accuracy (maior valor)
        )
    ]

    final_model = build_model_from_params(best_params_pso) # constroi a rede CNN com os valores lidos do json
    final_model.compile( # compila o modelo final
        optimizer=keras.optimizers.Adam(learning_rate=best_params_pso['lr']),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    print(f"A iniciar treino de {FINAL_EPOCHS} épocas (patience={FINAL_PATIENCE})...")
    
    history = final_model.fit( # treina o modelo e guarda no history para fazer o grafico da loss e acc mais tarde
        train_ds,
        validation_data=val_ds,
        epochs=FINAL_EPOCHS,
        callbacks=final_callbacks,
        verbose=1 # agora ja mostra o progresso do treino, visto que é so um treino e nao dezenas/centenas
    )
    
    print("\n" + "=" * 50)
    print("RE-TREINO FINAL CONCLUÍDO.")
    
    # Salva o historico do treino para fazer plot dos graficos de loss e accuracy
    with open(HISTORY_SAVE_PATH, "wb") as f:
        pickle.dump(history.history, f)
        
    print(f"\nModelo final guardado em: {MODEL_SAVE_PATH}")
    print(f"Histórico guardado em: {HISTORY_SAVE_PATH}")
    print("=" * 50)