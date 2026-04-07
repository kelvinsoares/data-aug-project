import os
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report

# =========================================================
# REPRODUTIBILIDADE
# =========================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# =========================================================
# CONFIG
# =========================================================
BASE_DIR = "/home/kevin/dataset_pdi/svd_melspec_aug"
IMG_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 25

# Escolha: "vgg16" ou "densenet121"
BACKBONE = "vgg16"

# =========================================================
# BACKBONE + PREPROCESSING
# =========================================================
if BACKBONE == "vgg16":
    from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
    base_cnn = VGG16(weights="imagenet", include_top=False, input_shape=(128, 128, 3))
elif BACKBONE == "densenet121":
    from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input
    base_cnn = DenseNet121(weights="imagenet", include_top=False, input_shape=(128, 128, 3))
else:
    raise ValueError("BACKBONE deve ser 'vgg16' ou 'densenet121'")

base_cnn.trainable = False  # começa congelado

# =========================================================
# DATA GENERATORS
# =========================================================
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_generator = train_datagen.flow_from_directory(
    f"{BASE_DIR}/train",
    target_size=IMG_SIZE,
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=True,
    seed=SEED
)

val_generator = val_datagen.flow_from_directory(
    f"{BASE_DIR}/validation",
    target_size=IMG_SIZE,
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

test_generator = test_datagen.flow_from_directory(
    f"{BASE_DIR}/test",
    target_size=IMG_SIZE,
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

# =========================================================
# MODELO
# =========================================================
inputs = Input(shape=(128, 128, 3))
x = base_cnn(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.4)(x)
outputs = Dense(1, activation="sigmoid")(x)

model = Model(inputs, outputs)

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =========================================================
# CALLBACKS
# =========================================================
checkpoint_path = f"best_{BACKBONE}.keras"

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=1e-6,
    verbose=1
)

checkpoint = ModelCheckpoint(
    checkpoint_path,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

# =========================================================
# TREINO
# =========================================================
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[early_stop, reduce_lr, checkpoint]
)

# =========================================================
# TESTE
# =========================================================
loss, acc = model.evaluate(test_generator, verbose=1)
print(f"\nAcurácia no teste: {acc:.4f}")

# =========================================================
# PREVISÕES
# =========================================================
y_pred_prob = model.predict(test_generator)
y_pred = (y_pred_prob > 0.5).astype(int).ravel()
y_true = test_generator.classes

# =========================================================
# MATRIZ DE CONFUSÃO
# =========================================================
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(5, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["saudavel", "patologia"],
    yticklabels=["saudavel", "patologia"]
)
plt.xlabel("Predito")
plt.ylabel("Real")
plt.title(f"Matriz de Confusão - {BACKBONE}")
plt.tight_layout()
plt.show()

# =========================================================
# RELATÓRIO
# =========================================================
print("\nRelatório de Classificação:")
print(classification_report(
    y_true,
    y_pred,
    target_names=["saudavel", "patologia"]
))

# =========================================================
# GRÁFICOS
# =========================================================
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="train")
plt.plot(history.history["val_accuracy"], label="val")
plt.title("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="val")
plt.title("Loss")
plt.legend()

plt.tight_layout()
plt.show()