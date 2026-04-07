import numpy as np
from PIL import Image
from pathlib import Path
import random
from tqdm import tqdm
import shutil

# =========================
# CONFIG
# =========================
INPUT_DIR = Path("/home/kevin/dataset_pdi/svd_melspec")
OUTPUT_DIR = Path("/home/kevin/dataset_pdi/svd_melspec_aug")

CLASSES = ["saudavel", "patologia"]

# =========================
# AUGMENTATIONS
# =========================

def time_mask(img_array):
    h, w = img_array.shape

    t = random.randint(10, 30)  # largura da máscara
    t0 = random.randint(0, w - t)

    img_array[:, t0:t0+t] = 0
    return img_array

def freq_mask(img_array):
    h, w = img_array.shape

    f = random.randint(10, 30)
    f0 = random.randint(0, h - f)

    img_array[f0:f0+f, :] = 0
    return img_array

# =========================
# PROCESSAMENTO
# =========================

def process_train():
    for cls in CLASSES:
        in_dir = INPUT_DIR / "train" / cls
        out_dir = OUTPUT_DIR / "train" / cls
        out_dir.mkdir(parents=True, exist_ok=True)

        files = list(in_dir.glob("*.png"))

        for img_path in tqdm(files, desc=f"train/{cls}"):
            img = Image.open(img_path).convert("L")
            img_array = np.array(img)

            # salvar original
            out_path = out_dir / img_path.name
            img.save(out_path)

            # time mask
            tm = time_mask(img_array.copy())
            Image.fromarray(tm).save(out_dir / f"{img_path.stem}_tm.png")

            # freq mask
            fm = freq_mask(img_array.copy())
            Image.fromarray(fm).save(out_dir / f"{img_path.stem}_fm.png")

# =========================
# COPIAR VAL E TEST
# =========================

def copy_split(split):
    for cls in CLASSES:
        src = INPUT_DIR / split / cls
        dst = OUTPUT_DIR / split / cls
        dst.mkdir(parents=True, exist_ok=True)

        for file in src.glob("*.png"):
            shutil.copy(file, dst / file.name)

# =========================
# MAIN
# =========================

def main():
    process_train()

    copy_split("validation")
    copy_split("test")

    print("\nAugmentation concluída!")

if __name__ == "__main__":
    main()
