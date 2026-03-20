# benchmark_streaming_single.py
import time
import tensorflow as tf

# ===== Ajusta só isto se quiseres =====
IMG_SIZE = 128        # 128 / 160 / 224 / 256
NUM_CLASSES = 5
BATCH = 64           # baixa se der OOM (ex.: 128, 64)
STEPS = 50             # passos por época (maior = mais pesado)
EPOCHS = 1
# =====================================

print("\n\nDevices visíveis:", tf.config.get_visible_devices(), "\n\n")

# Dataset sintético por batch (não guarda tudo em RAM)
def make_stream_ds(steps, batch, img, n_classes):
    ds = tf.data.Dataset.range(steps)

    def gen(_):
        images = tf.random.uniform((batch, img, img, 3), dtype=tf.float32)
        labels = tf.random.uniform((batch,), maxval=n_classes, dtype=tf.int32)
        return images, labels

    ds = ds.map(gen, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)

ds = make_stream_ds(STEPS, BATCH, IMG_SIZE, NUM_CLASSES)

def make_model():
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = inputs
    # 4 blocos conv pesadinhos
    for f in (64, 128, 256, 256):
        for _ in range(2):
            x = tf.keras.layers.Conv2D(f, 3, padding='same', use_bias=False)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    return model

model = make_model()
model.summary(line_length=120)

# Warm-up (compilação inicial)
print("\nWarming up...")
model.fit(ds.take(5), epochs=1, verbose=0)

# Benchmark
print(f"\nBenchmark: IMG={IMG_SIZE}  BATCH={BATCH}  STEPS={STEPS}  EPOCHS={EPOCHS}")
t0 = time.time()
model.fit(ds.take(STEPS), epochs=EPOCHS, verbose=1)
t1 = time.time()

elapsed = t1 - t0
steps_per_sec = (STEPS * EPOCHS) / elapsed
imgs_per_sec  = steps_per_sec * BATCH
print(f"\nTempo total: {elapsed:.2f}s | {steps_per_sec:.2f} steps/s | {int(imgs_per_sec)} imgs/s")
