from pathlib import Path
import numpy as np
import librosa
from PIL import Image
from tqdm import tqdm

# =========================
# CONFIGURACOES
# =========================
INPUT_ROOT = Path("/home/kevin/dataset_pdi/svd")
OUTPUT_ROOT = Path("/home/kevin/dataset_pdi/svd_melspec")

SPLITS = ["train", "validation", "test"]
CLASSES = ["saudavel", "patologia"]

# Mantem a taxa original da base (no seu caso, 50000 Hz)
SR = None

# Parametros do mel-spectrograma
N_MELS = 128
FMAX = 8000
N_FFT = 2048
HOP_LENGTH = 512

# Tamanho final da imagem para a CNN
TARGET_SIZE = (128, 128)

# Faixa dinamica usada na normalizacao em dB
TOP_DB = 80.0


# =========================
# FUNCAO: AUDIO -> IMAGEM
# =========================
def audio_to_mel_image(audio_path: Path) -> Image.Image:
    # Carrega o audio
    y, sr = librosa.load(audio_path, sr=SR, mono=True)

    if y.size == 0:
        raise ValueError("Audio vazio")

    # Gera o mel-spectrograma
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=0,
        fmax=FMAX,
        power=2.0
    )

    # Converte para escala logarItmica (dB)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Normaliza para a faixa [0, 1]
    mel_db = np.clip(mel_db, -TOP_DB, 0)
    mel_norm = (mel_db + TOP_DB) / TOP_DB

    # Converte para imagem em tons de cinza (0-255)
    img_array = (mel_norm * 255).astype(np.uint8)

    # Cria imagem PIL e redimensiona para entrada fixa da CNN
    img = Image.fromarray(img_array, mode="L")
    img = img.resize(TARGET_SIZE, Image.Resampling.BILINEAR)

    return img


# =========================
# PROCESSA UMA PASTA
# =========================
def process_split(split: str):
    for cls in CLASSES:
        in_dir = INPUT_ROOT / split / cls
        out_dir = OUTPUT_ROOT / split / cls
        out_dir.mkdir(parents=True, exist_ok=True)

        files = sorted(in_dir.glob("*.wav"))

        print(f"\nProcessando {split}/{cls} -> {len(files)} arquivos")

        for audio_path in tqdm(files, desc=f"{split}/{cls}"):
            out_path = out_dir / f"{audio_path.stem}.png"

            # Evita reprocessar se o arquivo ja existir
            if out_path.exists():
                continue

            try:
                img = audio_to_mel_image(audio_path)
                img.save(out_path)
            except Exception as e:
                print(f"Erro em {audio_path.name}: {e}")


# =========================
# MAIN
# =========================
def main():
    for split in SPLITS:
        process_split(split)

    print("\nConcluido. Espectrogramas salvos em:", OUTPUT_ROOT)


if __name__ == "__main__":
    main()
