# train_mlp.py

import tensorflow as tf, pickle
from tensorflow import keras
from tensorflow.keras import layers

# pre-processamento e normalizacao das imagens
def ds(path, training):
    d = keras.preprocessing.image_dataset_from_directory( # d é uma lista que guarda 2valores (imagem=x, etiqueta=y) | o x é uma matriz multidimensional (128, 128, 3), onde o '128, 128' é altura e largura de cada imagem e o 3 é o num de cores (RGB), dentro dessa matriz existem os valores da cor de cada pixel (p ex: [255, 0, 0] -> Vermelho) | já o y é a classe de cada imagem, como nós estamos a usar 'label_mode'="categorical", em vez de o y guardar o nome da classe, guarda num vetor One-Hot, onde basicamente tem o numero de colunas = ao numero de classes que existe (p ex: [1,0,0,0,0])), e a classe correta da imagem aparece com o número '1', no exemplo anterior a imagem seria da classe "convertible", sendo a 1a classe das 5 por ordem alfabética
        path, image_size=(128, 128), batch_size=32,
        label_mode="categorical", shuffle=training
    )
    return d.map(lambda x,y: (tf.cast(x, tf.float32)/255.0, y)).prefetch(tf.data.AUTOTUNE) # passa os valores das cores dos pixels (daquela matriz(x) em d para float e divide por 255 (valor max do RGB) para normalizar os dados (o intervalo fica entre 0.0 e 1.0) o que faz com que as redes neuronais treinem melhor | O comando .prefetch(tf.data.AUTOTUNE) no train.py garante que o próximo lote de dados está sempre pronto na GPU/CPU enquanto o lote atual está a ser processado, maximizando o desempenho do treino

# carregar dataset
train_ds = ds("../../dataset/train", True)
val_ds = ds("../../dataset/validation", False)

# arquitetura do modelo MLP
model = keras.Sequential([
    layers.Input((128, 128, 3)),
    layers.Flatten(),              

    layers.Dense(32, activation="relu"), layers.BatchNormalization(),
    layers.Dense(64, activation="relu"), layers.BatchNormalization(),
    layers.Dense(128, activation="relu"), layers.BatchNormalization(),

    layers.Dropout(0.5), # Desliga 50% dos neurónios aleatoriamente durante o treino para evitar o overfitting
    layers.Dense(5, activation="softmax") # 5 classes
])

# earlyStopping
earlyStopping = keras.callbacks.EarlyStopping( 
    monitor="val_loss",       # Keras analisa através da loss da validacao (tmb pode analisar através da "val_accuracy")
    patience=10,              # nº de épocas sem melhorar até parar
    restore_best_weights=True # no fim de cada época é guardado na memória RAM o melhor modelo (c melhor accyracy). Logo, no fim das épocas todas, caso não tenha feito early stopping, o Keras restaura para a memória RAM o modelo com melhor accuracy, de forma a 'model.save(MODEL_PATH)' salvar esse modelo(p.ex: epoca: 195, val_accuracy: 0.69) e não o último que tinha sido lido(p.ex: epoca: 200, val_accuracy: 0.61)    
)

# parametros de treino
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"])

# treino
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=100,
    callbacks=[earlyStopping] # No final de cada época, o Keras chama a função 'earlyStopping'
)

# caso o treino pare pelo earlyStopping
if earlyStopping.stopped_epoch > 0:
    print(f"\nTreino parado automaticamente por EarlyStopping (sem melhoria nas últimas {earlyStopping.patience} épocas).")

# salva o melhor modelo do treino e o historico do treino para fazer plot dos graficos de loss e accuracy
model.save("bestMLP.keras")
with open("historyMLP.pkl", "wb") as f:
    pickle.dump(history.history, f)

print("\nModelo guardado em: bestMLP.keras")
print("Histórico guardado em: historyMLP.pkl")
