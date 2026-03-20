import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # parar tirar mensagens do back-end em C++ do tensorflow
import tensorflow as tf, pickle

import numpy as np
import time
from tensorflow import keras
from tensorflow.keras import layers
from SwarmPackagePy import gwo
import json       
import threading  
import warnings

# parar tirar mensagens do lado do python do tensorflow
tf.get_logger().setLevel('ERROR')
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

SAVE_FILE = "gwo_best_result.json" # no final de cada treino feito pelo otimizador, vai ler este ficheiro que tem guardado a melhor acc (com os parametros que foram usados nesse treino), se o treino atual tiver melhor acc, reescreve-ve neste ficheiro a aac atual e os parametros que foram usados para este treino
JSON_LOCK = threading.Lock()

# pre-processamento e normalizacao das imagens
def ds(path, training):
    d = keras.preprocessing.image_dataset_from_directory( # d é uma lista que guarda 2valores (imagem=x, etiqueta=y) | o x é uma matriz multidimensional (128, 128, 3), onde o '128, 128' é altura e largura de cada imagem e o 3 é o num de cores (RGB), dentro dessa matriz existem os valores da cor de cada pixel (p ex: [255, 0, 0] -> Vermelho) | já o y é a classe de cada imagem, como nós estamos a usar 'label_mode'="categorical", em vez de o y guardar o nome da classe, guarda num vetor One-Hot, onde basicamente tem o numero de colunas = ao numero de classes que existe (p ex: [1,0,0,0,0])), e a classe correta da imagem aparece com o número '1', no exemplo anterior a imagem seria da classe "convertible", sendo a 1a classe das 5 por ordem alfabética
        path, image_size=(128, 128), batch_size=32,
        label_mode="categorical", shuffle=training
    )
    return d.map(lambda x,y: (tf.cast(x, tf.float32)/255.0, y)).prefetch(tf.data.AUTOTUNE) # passa os valores das cores dos pixels (daquela matriz(x) em d para float e divide por 255 (valor max do RGB) para normalizar os dados (o intervalo fica entre 0.0 e 1.0) o que faz com que as redes neuronais treinem melhor | O comando .prefetch(tf.data.AUTOTUNE) no train.py garante que o próximo lote de dados está sempre pronto na GPU/CPU enquanto o lote atual está a ser processado, maximizando o desempenho do treino

print("A carregar datasets de treino e validação...")
train_ds = ds("../../dataset/train", True)
val_ds = ds("../../dataset/validation", False)
print("Datasets carregados.")

# transforma os valores vindos do algoritmo GWO (entre 0.0 e 1.0 (limites definidos em lb (low bound) e ub(upper bound))) em hiperparâmetros reais e utilizáveis pela CNN
def decode_params(params_0_1): 
    
    # numero de filtros de cada bloco (estático, nem conta como hiperparametro a otimizar pelo algoritmo)
    f1, f2, f3 = 32, 64, 128 
    
    # 1º Hiperparametro categórico/discreto - kernel (janela que a CNN olha de cada vez). Duas opcoes: (3,3) -> 3x3 pixeis | (5,5) -> 5x5 pixeis
    kernel_size = 3 if params_0_1[0] < 0.5 else 5 # params_0_1[0] escolhe o kernel. < 0.5 = (3,3), >= 0.5 = (5,5)
    
    # 2º Hiperparametro contínuo - Learning Rate(lr (taxa de aprendizagem) - controla o tamanho do passo que o otimizador da funcao de treino (Adam) dá ao atualizar os pesos da rede durante o treino)
    lr_min, lr_max = 1e-4, 1e-2 # varia de 0.0001 a 0.01
    lr = 10 ** (np.log10(lr_min) + params_0_1[1] * (np.log10(lr_max) - np.log10(lr_min))) # gera um lr no intervalo a cima usando escala logarítmica

    # 3º, 4º e 5º Hiperparametros contínuos - Dropouts(define a fração de neurónios que são desligados aleatoriamente durante o treino)
    d1     = 0.10 + params_0_1[2] * (0.50 - 0.10)  # Dropout no final do bloco 1
    d2_3   = 0.20 + params_0_1[3] * (0.60 - 0.20)  # Dropout no final dos blocos 2 e 3 (mesmo valor)
    d_dense= 0.30 + params_0_1[4] * (0.70 - 0.30)  # Dropout no classificador 
    
    params_dict = { # retorna uma especie de array com o valor de cada parametro em cada coluna do array
        "f1": f1,
        "f2": f2,   
        "f3_dense": f3,
        "kernel": int(kernel_size),
        "lr": float(lr),
        "d1": float(d1),
        "d2_3": float(d2_3),
        "d_dense": float(d_dense)
    }
    return params_dict

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

# treina varias vezes o modelo (durante a otimizacao)
def avaliar_modelo(params_0_1): # em 'params_0_1' estão os valores entre 0 e 1 criados e enviados pelo algoritmo GWO
    try:
        p = decode_params(params_0_1) # tenta transformar os valore entre 0 e 1 para valores reais (ficam no array 'p')
    except Exception as e:
        print(f"Erro ao descodificar parâmetros: {e}")
        return 1.0 # Pior fitness
        
    print(f"\n[TESTE] K:{p['kernel']} | F1:{p['f1']} | F2:{p['f2']} | F3_D:{p['f3_dense']} | LR:{p['lr']:.5f} | D1:{p['d1']:.2f} | D2_3:{p['d2_3']:.2f} | D_D:{p['d_dense']:.2f}")

    try:
        model = build_model_from_params(p) # cria o modelo com a arquitetura CNN ja c os valores transformados vindos do GWO
        
        earlyStopping = keras.callbacks.EarlyStopping( 
            monitor="val_loss",       
            patience=10, # 7 5 10
            restore_best_weights=True 
        )

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=p['lr']),
            loss="categorical_crossentropy", 
            metrics=["accuracy"])

        history = model.fit( # modelo é treinado (80% treino)
            train_ds,
            validation_data=val_ds,
            epochs=70, # 70 50 100
            callbacks=[earlyStopping],
            verbose=0 
        )   

        loss, accuracy = model.evaluate(val_ds, verbose=0) # modelo é avaliado (10% validacao)
        print(f"[RESULTADO] Accuracy: {accuracy:.4f} | Loss: {loss:.4f} | Fitness: {1.0 - accuracy:.4f}")

        # chekar para ver se o que está guardado no json é melhor do que este que deu agora
        novo_resultado = {
            "melhor_accuracy": accuracy,
            "parametros_descodificados": p,
            "parametros_raw_0_1": list(params_0_1)
        }

        with JSON_LOCK:
            score_antigo = -1.0 # para caso nao exista o ficheiro, por defeito é -1.0
            if os.path.exists(SAVE_FILE):
                try:
                    with open(SAVE_FILE, 'r') as f:
                        dados_antigos = json.load(f)
                        score_antigo = dados_antigos.get("melhor_accuracy", -1.0) # caso nao exista nenhum 'melhor_accuracy' no json, fica -1.0 por defeito
                except Exception as e:
                    print(f"[AVISO] Não foi possível ler {SAVE_FILE}: {e}")
            
            if accuracy > score_antigo:
                print(f"[CHECKPOINT] Novo melhor resultado! Acc: {accuracy:.4f} (anterior: {score_antigo:.4f}). A salvar...")
                try:
                    with open(SAVE_FILE, 'w') as f:
                        json.dump(novo_resultado, f, indent=4) # escreve no ficheiro variavel 'f', o resultado que se obteve agora com formatacao ident=4 (4 espacos)
                except Exception as e:
                    print(f"[ERRO] Não foi possível salvar {SAVE_FILE}: {e}")
            else:
                print(f"[CHECKPOINT] Resultado (Acc: {accuracy:.4f}) não superou o melhor (Acc: {score_antigo:.4f}).")
        
        keras.backend.clear_session() # limpa toda a sessão atual do TensorFlow/Keras (limpa memória de GPU, RAM, etc)
        del model # apaga a variavel model da memoria do python
        
        return 1.0 - accuracy # Como fitness é a metrica que o GWO tenta minimizar e a metrica que se está a usar no treino é a accuracy (que se tende a maximizar), tem que se fazer 1-aac de forma a obter o valor minimo (minimizar)

    except Exception as e:
        print(f"[ERRO NO TREINO] {e}")
        keras.backend.clear_session() # limpa toda a sessão atual do TensorFlow/Keras (limpa memória de GPU, RAM, etc)
        return 1.0 

dim = 5 # numero de hiperparametros a otimizar pelo otimizador
lb = np.array([0.0] * dim) # array com os valores de  lowbound  para cada hiperparametro ([0.0, 0.0, 0.0, 0.0, 0.0]) 
ub = np.array([1.0] * dim) # array com os valores de upperbound para cada hiperparametro ([1.0, 1.0, 1.0, 1.0, 1.0]) 

pop_size = 15 # 10 agentes, neste caso lobos
iterations = 13 # 10 iteracoes

# executa o otimizador do Gray Wolf Optimization (GWO)
print("--- OTIMIZAÇÃO DE HIPERPARÂMETROS (PASSO 1/2) ---")
print(f"Dimensão: {dim}, População: {pop_size}, Iterações: {iterations}")
# modelo proxy - modelo CNN pequeno/rápido treinado apenas durante a otimização (umas das combinacoes geradas pelo otimizador)
print(f"Total de modelos 'proxy' a treinar: {pop_size * iterations}\n")

print(f"A iniciar GWO...")
start_gwo = time.time() # para cronometrar o tempo da otimizacao
gwo(pop_size, avaliar_modelo, lb, ub, dim, iterations) # otimizacao
end_gwo = time.time() # para terminar o cronometro

print("\n" + "=" * 50)
print(f"OTIMIZAÇÃO GWO COMPLETA em {(end_gwo - start_gwo) / 3600:.2f} horas")

# le o melhor resultado do JSON para reetreinar o modelo com os melhores parametros encontrados na otimizacao
try:
    with open(SAVE_FILE, 'r') as f:
        dados_finais = json.load(f)
    best_params_gwo = dados_finais["parametros_descodificados"]
    print(f"  Melhor Accuracy (Validação) encontrada: {dados_finais['melhor_accuracy']:.6f}")
    print("\n  Melhor conjunto de Hiperparâmetros (lido do JSON):")
    for key, value in best_params_gwo.items():
        print(f"    {key}: {value:.6f}" if isinstance(value, float) else f"    {key}: {value}")
except Exception as e:
    print(f"[ERRO] Não foi possível ler o ficheiro de resultados {SAVE_FILE}: {e}")
    print("Re-treino final cancelado.")
    best_params_gwo = None
print("=" * 50)

# re-treina o modelo final com os melhores valores dos hiperparametros encontrados pelo otimizador. Só executa o treino se os parâmetros foram carregados com sucesso
if best_params_gwo:
    print("\n--- RE-TREINO FINAL (PASSO 2/2) ---")
    print("A treinar o modelo final com os melhores parâmetros...")

    MODEL_SAVE_PATH = "final_model_gwo.keras"

    # callbacks para o treino final
    final_callbacks = [
        keras.callbacks.EarlyStopping( # callback que serve APENAS PARA parar o treino se não melhorar
            monitor="val_loss", # controla pela loss
            patience=10, # nº de épocas sem melhorar até parar
            restore_best_weights=True # no final de cada epoca salva na RAM o melhor modelo se for o com melhor accuracy para no final do treino fazer restauro do modelo e salvar(model.save), mas neste caso nao se está a fazer save dessa forma, mas sim com o callback 'ModelCheckpoint' para ser mais seguro, que salva mesmo já o ficheiro .keras ao final da epoca (caso seja o com melhor valor de acc) antes que dê problema com a RAM durante o treino
        ),
        keras.callbacks.ModelCheckpoint( # callback que serve para salvar o melhor modelo no ficheiro .keras a cada epoca (se for melhor que o anterior melhor)
            filepath=MODEL_SAVE_PATH, # nome do ficheiro
            monitor="val_loss", # controla pela loss
            save_best_only=True # no final de cada epoca salva o ficheiro .keras com o modelo se for o com melhor accuracy (maior valor)
        )
    ]
    
    final_model = build_model_from_params(best_params_gwo) # constroi a rede CNN com os valores lidos do json
    final_model.compile( # compila o modelo final
        optimizer=keras.optimizers.Adam(learning_rate=best_params_gwo['lr']),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    print(f"A iniciar treino de 100 épocas (patience=10)...")
    
    history = final_model.fit( # treina o modelo e guarda no history para fazer o grafico da loss e acc mais tarde
        train_ds,
        validation_data=val_ds,
        epochs=100,
        callbacks=final_callbacks,
        verbose=1 # agora ja mostra o progresso do treino, visto que é so um treino e nao dezenas/centenas
    )
    
    print("\n" + "=" * 50)
    print("RE-TREINO FINAL CONCLUÍDO.")
    
    # salva o historico do treino para fazer plot dos graficos de loss e accuracy
    with open("history_gwo.pkl", "wb") as f:
        pickle.dump(history.history, f)
        
    print("\nModelo final guardado em: final_model_gwo.keras")
    print("Histórico guardado em: history_gwo.pkl")
    print("=" * 50)
    