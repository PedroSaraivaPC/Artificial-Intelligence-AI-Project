import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

file = "700"

# ============================================
# LER CSV DO OTIMIZADOR (ex.: train_700)
# ============================================
df = pd.read_csv(f"gwo_results_resnet_train{file}.csv")

# ============================================
# 3. Heatmap (units × dropout × accuracy)
# ============================================
bins = np.linspace(df["dropout"].min(), df["dropout"].max(), 6)
labels = [f"{round(bins[i],2)}–{round(bins[i+1],2)}" for i in range(len(bins)-1)]

df["dropout"] = pd.cut(df["dropout"], bins=bins, labels=labels, include_lowest=True)

pivot = df.pivot_table(values="val_accuracy",
                       index="dropout",
                       columns="units",
                       aggfunc="mean")

plt.figure(figsize=(8,6))
x = sns.heatmap(pivot, annot=True, cmap="viridis", fmt=".3f")

colorbar = x.collections[0].colorbar
colorbar.set_label("Accuracy média", fontsize=11)

plt.title(f"Accuracy vs N.º de Neurónios vs Dropout (GWO - {file} imagens por classe)")
plt.xlabel("Número de neurónios")
plt.ylabel("Dropout")
plt.tight_layout()
plt.show()







# COMPARAÇÃO DOS 3 OTIMIZADORES (GWO vs GRID vs RANDOM)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ========================================================
# LER OS 3 CSVs DOS OTIMIZADORES (APENAS PARA train_700)
# ========================================================
files = {
    "GWO": "gwo_results_resnet_train700.csv",
    "Grid": "grid_results_resnet_train700.csv",
    "Random": "random_results_resnet_train700.csv",
}

dfs = {}
for name, path in files.items():
    df = pd.read_csv(path)
    #df = df.sort_values("val_accuracy").reset_index(drop=True)
    dfs[name] = df

# ========================================================
# 1. Evolução da accuracy (100 modelos de cada otimizador)
# ========================================================
plt.figure(figsize=(12,6))

for name, df in dfs.items():
    plt.plot(df["val_accuracy"], label=name, linewidth=1)

plt.title(f"Comparação da Evolução da Accuracy – GWO vs Grid vs Random ({file} imagens por classe)")
plt.xlabel("Modelo (ordenado por accuracy)")
plt.ylabel("Accuracy de validação")
plt.legend()
plt.grid()
plt.show()

# ========================================================
# 2. Distribuição KDE das accuracies (comparação direta)
# ========================================================
plt.figure(figsize=(12,6))
for name, df in dfs.items():
    sns.kdeplot(df["val_accuracy"], label=name, linewidth=2)

plt.title(f"Distribuição das Accuracies – GWO vs Grid vs Random ({file} imagens por classe)")
plt.xlabel("Accuracy")
plt.ylabel("Densidade / Frequência")
plt.legend()
plt.show()