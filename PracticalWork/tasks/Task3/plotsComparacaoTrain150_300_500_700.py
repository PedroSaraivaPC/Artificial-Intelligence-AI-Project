import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# comparacao individual

# ------------------------------------------------------
# LER FICHEIRO CSV
# ------------------------------------------------------
df = pd.read_csv("gwo_results_resnet_train700.csv")

# ------------------------------------------------------
# 1. Scatter Plot -> Accuracy em função do número de épocas (cada ponto representa um modelo dos 140 que houveram)
# ------------------------------------------------------
plt.figure(figsize=(10,6))
plt.scatter(df["epochs_treinadas"], df["val_accuracy"], alpha=0.7)
plt.title("Accuracy em função do número de épocas (train_700)")
plt.xlabel("Épocas treinadas (EarlyStopping)")
plt.ylabel("Accuracy de validação")
plt.grid()
plt.show()

# ------------------------------------------------------
# 2. Heatmap -> accuracy média vs dropout vs units
# ------------------------------------------------------

# 2.1. Criar bins para o dropout (por exemplo 5 intervalos)
bins = np.linspace(df["dropout"].min(), df["dropout"].max(), 6)
labels = [f"{round(bins[i],2)}–{round(bins[i+1],2)}" for i in range(len(bins)-1)]

df["dropout_bin"] = pd.cut(df["dropout"], bins=bins, labels=labels, include_lowest=True)

# 2.2. Criar tabela pivotada com a média das accuracies por bin
pivot = df.pivot_table(values="val_accuracy",
                       index="dropout_bin",
                       columns="units",
                       aggfunc="mean")

# 2.3. Heatmap com bins
plt.figure(figsize=(8,6))
sns.heatmap(pivot, annot=True, cmap="viridis", fmt=".3f")
plt.title("Heatmap Accuracy – train_700")
plt.xlabel("Units")
plt.ylabel("Dropout (intervalos)")
plt.show()



# comparacao das 4 pastas num grafico
import glob
import pandas as pd
import matplotlib.pyplot as plt

files = {
    "train_150": "gwo_results_resnet_train150.csv",
    "train_300": "gwo_results_resnet_train300.csv",
    "train_500": "gwo_results_resnet_train500.csv",
    "train_700": "gwo_results_resnet_train700.csv",
}

plt.figure(figsize=(12,6))

for label, path in files.items():
    df = pd.read_csv(path)
    plt.plot(df["val_accuracy"].reset_index(drop=True), label=label)

plt.title("Evolução da Accuracy para cada pasta")
plt.xlabel("Modelo avaliado")
plt.ylabel("Accuracy de validação")
plt.legend()
plt.grid()
plt.show()
