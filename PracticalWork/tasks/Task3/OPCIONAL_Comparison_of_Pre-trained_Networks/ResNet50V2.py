# 02:26.71
import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # parar tirar mensagens do back-end em C++ do tensorflow

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50V2
import pickle

# parar tirar mensagens do lado do python do tensorflow
tf.get_logger().setLevel('ERROR')
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# =====================================================================
#  PRE-PROCESSAMENTO (mantém igual ao teu estilo)
# =====================================================================
def ds(path, training):
    d = keras.preprocessing.image_dataset_from_directory(
        path,
        image_size=(224, 224),
        batch_size=32,
        label_mode="categorical",
        shuffle=training
    )

    return d.map(lambda x,y: (tf.cast(x, tf.float32)/255.0, y)) \
            .prefetch(tf.data.AUTOTUNE)

print("\nA carregar datasets...")
train_ds = ds("../../dataset/train_700", True)
val_ds   = ds("../../dataset/validation", False)
print("Datasets carregados com sucesso!")

# =====================================================================
#  CARREGAR ResNet50V2 PRÉ-TREINADA (ImageNet)
# =====================================================================
base_model = ResNet50V2(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3)
)

# CONGELAR backbone
base_model.trainable = False

# =====================================================================
#  DEFINIR TOP-LAYER (rede densa personalizada)
# =====================================================================
inputs = keras.Input(shape=(224, 224, 3))

x = base_model(inputs, training=False)      # manter BN estável
x = layers.GlobalAveragePooling2D()(x)

# top-layer fixa (baseline)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.4)(x)

outputs = layers.Dense(5, activation="softmax")(x)

model = keras.Model(inputs, outputs)

# =====================================================================
#  COMPILAR MODELO
# =====================================================================
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nResumo do modelo:")
print(model.summary())

# =====================================================================
#  TREINO BASELINE
# =====================================================================
print("\nA iniciar treino baseline...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=25,
    verbose=1
)

print("\nTreino concluído!")

# =====================================================================
#  GUARDAR MODELO FINAL
# =====================================================================
model.save("ResNet50V2.keras")
print("\nModelo guardado como ResNet50V2.keras")

# =====================================================================
#  GUARDAR HISTÓRICO
# =====================================================================

with open("history_ResNet50V2.pkl", "wb") as f:
    pickle.dump(history.history, f)

print("Histórico guardado!\n")

