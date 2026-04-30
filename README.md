# Audio-to-Image CNN Classifier and Data Augmentation

## Sobre o Projeto
Este repositório contém o código-fonte de um projeto de Processamento Digital de Imagens. O objetivo principal do projeto é o processamento e classificação de sinais de áudio utilizando técnicas avançadas de Visão Computacional e Data augmentation. 

Para isso, o pipeline converte amostras de áudio em Espectrogramas de Mel (representações visuais do som) e utiliza Redes Neurais Convolucionais (CNNs) para extração de características e classificação, além de técnicas de Data Augmentation para melhorar as métricas do modelo.

## ⚙️ Destaques Técnicos e Pipeline
- **Processamento Digital de Imagens:** Conversão de séries temporais (áudio) em Espectrogramas de Mel.
- **Deep Learning:** Arquitetura e treinamento de CNNs customizadas para classificação das imagens geradas.
- **Data Augmentation e Métricas:** Aplicação de *Data Augmentation* para expandir, balancear e diversificar a base de treinamento. Essa etapa foi fundamental para evitar *overfitting*, melhorar significativamente as métricas de avaliação do modelo (como acurácia e F1-score) e garantir uma maior capacidade de generalização.
- **Base de dados utilizada:** SVD (Saarbruecken Voice Database)
  
## 🛠️ Tecnologias Utilizadas
- **Linguagem:** Python
- **Deep Learning:** TensorFlow/Keras
- **Processamento de Áudio e Imagem:** Librosa, OpenCV, PIL, NumPy, Pandas
- **Ambiente:** Linux
