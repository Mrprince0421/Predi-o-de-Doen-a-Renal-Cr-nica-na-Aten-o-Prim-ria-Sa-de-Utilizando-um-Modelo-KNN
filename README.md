# Predição de Doença Renal Crônica (KNN do Zero)

Este projeto implementa um modelo de aprendizado de máquina para classificação e triagem de **Doença Renal Crônica (DRC)** utilizando o *Chronic Kidney Disease Dataset* (UCI ID: 336).

O algoritmo de classificação (**KNN - K-Nearest Neighbors**), o pré-processamento dos dados e a validação cruzada ($5\text{-Fold}$) foram **desenvolvidos do zero em Python**, sem o uso de bibliotecas de modelagem prontas (como `scikit-learn`).

---

## 🚀 Como Rodar o Projeto

### 1. Requisitos Prévia
Certifique-se de ter o Python instalado na versão 3.10 ou superior.

### 2. Instalar as Dependências
No terminal, instale os pacotes necessários para download da base e manipulação de matrizes:

```bash
pip install ucimlrepo pandas numpy
```

### 3. Executar o Código
Rode o script principal para treinar o modelo e visualizar as métricas ($98,75\%$ de acurácia média):

```bash
python main.py
```