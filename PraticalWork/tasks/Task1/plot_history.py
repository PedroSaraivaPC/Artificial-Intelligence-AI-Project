# plot_history.py

import pickle
import matplotlib.pyplot as plt

# lê o ficheiro com o historico do treino
with open("history.pkl", "rb") as f:
    history = pickle.load(f)

print("Métricas disponíveis no histórico:", list(history.keys()))

# grafico loss
plt.figure(figsize=(6,4))
plt.plot(history["loss"], label="Train")
plt.plot(history["val_loss"], label="Validation")
plt.title("Loss Evolution")
plt.xlabel("Ephocs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# grafico accuracy
plt.figure(figsize=(6,4))
plt.plot(history["accuracy"], label="Train")
plt.plot(history["val_accuracy"], label="Validation")
plt.title("Accuracy Evolution")
plt.xlabel("Ephocs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
