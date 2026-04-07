import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# =========================
# CONFIGURAÇÕES
# =========================

BASE_DIR = "/home/kevin/dataset_pdi/svd_melspec"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 15

# =========================
# GERADORES
# =========================

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
f"{BASE_DIR}/train",
target_size=IMG_SIZE,
color_mode="grayscale",
batch_size=BATCH_SIZE,
class_mode="binary")

val_generator = val_datagen.flow_from_directory(
f"{BASE_DIR}/validation",
target_size=IMG_SIZE,
color_mode="grayscale",
batch_size=BATCH_SIZE,
class_mode="binary")

test_generator = test_datagen.flow_from_directory(
f"{BASE_DIR}/test",
target_size=IMG_SIZE,
color_mode="grayscale",
batch_size=BATCH_SIZE,
class_mode="binary",
shuffle=False)

# =========================
# MODELO CNN
# =========================

model = Sequential([
Conv2D(32, (3,3), activation='relu', input_shape=(128,128,1)),
MaxPooling2D(2,2),
Conv2D(64, (3,3), activation='relu'),
MaxPooling2D(2,2),
Flatten(),
Dense(64, activation='relu'),
Dropout(0.5),
Dense(1, activation='sigmoid')])

# =========================
# COMPILAR
# =========================

model.compile(
optimizer='adam',
loss='binary_crossentropy',
metrics=['accuracy'])

model.summary()

# =========================
# TREINAMENTO
# ========================

history = model.fit(
train_generator,
validation_data=val_generator,
epochs=EPOCHS)

# =========================
# AVALIAÇÃO
# =========================

loss, acc = model.evaluate(test_generator)

print(f"\nAcurácia no teste: {acc:.4f}")

# =========================
# GRÁFICOS
# =========================

plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.title("Acurácia")
plt.legend()
plt.show()