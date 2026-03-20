# Best acc: 0.913 :/

import numpy as np, tensorflow as tf, matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, accuracy_score

# pre-processamento e normalizacao das imagens    
def ds_and_names(path):
    d = keras.preprocessing.image_dataset_from_directory( # d é uma lista que guarda 2valores (imagem=x, etiqueta=y) | o x é uma matriz multidimensional (128, 128, 3), onde o '128, 128' é altura e largura de cada imagem e o 3 é o num de cores (RGB), dentro dessa matriz existem os valores da cor de cada pixel (p ex: [255, 0, 0] -> Vermelho) | já o y é a classe de cada imagem, como nós estamos a usar 'label_mode'="categorical", em vez de o y guardar o nome da classe, guarda num vetor One-Hot, onde basicamente tem o numero de colunas = ao numero de classes que existe (p ex: [1,0,0,0,0])), e a classe correta da imagem aparece com o número '1', no exemplo anterior a imagem seria da classe "convertible", sendo a 1a classe das 5 por ordem alfabética
        path, image_size=(224, 224), batch_size=32,
        label_mode="categorical", shuffle=False
    )
    names = d.class_names             
    d = d.map(lambda x,y: (tf.cast(x, tf.float32)/255.0, y)).prefetch(tf.data.AUTOTUNE) # passa os valores das cores dos pixels (daquela matriz(x) em d para float e divide por 255 (valor max do RGB) para normalizar os dados (o intervalo fica entre 0.0 e 1.0) o que faz com que as redes neuronais treinem melhor | O comando .prefetch(tf.data.AUTOTUNE) no train.py garante que o próximo lote de dados está sempre pronto na GPU/CPU enquanto o lote atual está a ser processado, maximizando o desempenho do treino
    return d, names

# carregar dataset
test_ds, class_names = ds_and_names("../../dataset/test")
model = keras.models.load_model("final_model.keras")

# previsoes
proba = model.predict(test_ds) # proba é uma matriz(nºde imagens por nº classes), onde com o modelo salvo no treino, ele prevê as classes nas imagens de teste (imagem 1 - [0.01, 0.03, 0.90, 0.05, 0.01], neste exemplo preveu a imagem ser mais um Hathcback do que o resto) 
y_pred = proba.argmax(1) # y_pred fica com um vetor, uma coluna por imagem, onde cada valor vai ser o índice da classe (0, 1, 2, 3 ou 4 (5 classes)) com a maior probabilidade calculada na linha de cima, neste caso o indíce 2 (o 1 é para ser no eixo dessas probabilidades, se fosse 0 era no eixo das imagens)
y_true = np.concatenate([y.numpy().argmax(1) for _,y in test_ds]) # y_true fica com um vetor, uma coluna por imagem, onde cada valor vai ser o índice da classe correta (basicamente a mesma coisa que a linha de cima só que em vez de tirar do 'proba'(previsoes de cada imagem) retira do y do dataset d do pre-processamento(y = vetor one-hot),que é o rotulo/etiqueta(onde está guardado a classe correta de cada imagem))

# matriz de confusão
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,5))
plt.imshow(cm, cmap="Purples")
plt.title("Confusion matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks(range(len(class_names)), class_names, rotation=30)
plt.yticks(range(len(class_names)), class_names)
for i in range(len(class_names)):
    for j in range(len(class_names)):
        plt.text(j, i, cm[i,j], ha="center", va="center", color="white" if cm[i,j] > cm.max()/2 else "black")
plt.tight_layout()
plt.show()

# accuracy e report de classificacao (inclui: precisao, recall (sensibilidade) e f1-score(f-measure))
print("\nAccuracy:", accuracy_score(y_true, y_pred))
print("\nRelatório (precisão, recall(sensibilidade), f-measure):")
print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

# especificidade por classe -> TN / (TN + FP)
specs=[]
for i in range(len(class_names)):
    TP=cm[i,i]; 
    FP=cm[:,i].sum()-TP; # para ficar só com os erros nessa coluna
    FN=cm[i,:].sum()-TP; # para ficar só com os erros dessa linha
    TN=cm.sum()-(TP+FP+FN)
    specs.append(TN/(TN+FP) if (TN+FP)>0 else 0.0)
print("Especificidade por classe:")
for classe,valor in zip(class_names,specs): print(f"  {classe}: {valor:.4f}")

# AUC
y_true_1h = np.eye(len(class_names))[y_true]
auc_macro = roc_auc_score(y_true_1h, proba, average="macro", multi_class="ovr")
print("\nMacro AUC (OvR):", round(auc_macro,4))
