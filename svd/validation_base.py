import os
import librosa
import numpy as np

def validar_pasta(path):
    duracoes = []
    sample_rates = []
    erros = 0

    arquivos = [f for f in os.listdir(path) if f.endswith(".wav")]

    for arquivo in arquivos:
        caminho = os.path.join(path, arquivo)
        try:
            y, sr = librosa.load(caminho, sr=None)

            duracao = len(y) / sr
            duracoes.append(duracao)
            sample_rates.append(sr)

        except Exception:
            print(f"Erro ao abrir: {arquivo}")
            erros += 1

    print("\n========================")
    print(f"Pasta: {path}")
    print(f"Total arquivos: {len(arquivos)}")
    print(f"Erros: {erros}")
    print(f"Duração média: {np.mean(duracoes):.2f}s")
    print(f"Duração min/max: {np.min(duracoes):.2f} / {np.max(duracoes):.2f}")
    print(f"Sample rates únicos: {set(sample_rates)}")
    
validar_pasta("train/saudavel")
validar_pasta("train/patologia")

validar_pasta("validation/saudavel")
validar_pasta("validation/patologia")

validar_pasta("test/saudavel")
validar_pasta("test/patologia")
