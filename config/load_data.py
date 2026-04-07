from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Caminho da base
BASE_DIR = "/home/kevin/dataset_pdi/svd_melspec"

IMG_SIZE = (128, 128)
BATCH_SIZE = 32

# =========================
# TREINO (com augmentation depois)
# =========================
train_datagen = ImageDataGenerator(
    rescale=1./255
)

train_generator = train_datagen.flow_from_directory(
    f"{BASE_DIR}/train",
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

# =========================
# VALIDACAO
# =========================
val_datagen = ImageDataGenerator(rescale=1./255)

val_generator = val_datagen.flow_from_directory(
    f"{BASE_DIR}/validation",
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

# =========================
# TESTE
# =========================
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    f"{BASE_DIR}/test",
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

print(train_generator.class_indices)

x, y = next(train_generator)
print(x.shape, y.shape)
