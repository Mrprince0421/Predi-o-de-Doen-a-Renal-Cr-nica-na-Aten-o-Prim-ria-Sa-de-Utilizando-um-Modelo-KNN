from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np


chronic_kidney_disease = fetch_ucirepo(id=336)
X = chronic_kidney_disease.data.features
y = chronic_kidney_disease.data.targets


df = X.copy()
df['class'] = y
df['class'] = df['class'].astype(str).str.strip().str.replace('\t', '')
df['class'] = df['class'].map({'ckd': 1, 'notckd': 0, '1': 1, '0': 0})
df = df.dropna(subset=['class'])

X = df.drop(columns=['class'])
y = df['class'].values


for col in X.columns:
    X[col] = X[col].astype(str).str.strip().str.replace('\t', '')
    X[col] = X[col].replace(['?', 'nan', 'NaN', ''], np.nan)

    temp_num = pd.to_numeric(X[col], errors='coerce')

    if temp_num.notna().sum() > (len(X) * 0.5):
        X[col] = temp_num
        mediana = X[col].median()
        X[col] = X[col].fillna(mediana)
    else:
        moda = X[col].mode()[0] if not X[col].mode().empty else '0'
        X[col] = X[col].fillna(moda)
        unicos = X[col].unique()
        mapeamento = {val: i for i, val in enumerate(unicos)}
        X[col] = X[col].map(mapeamento)

X_array = X.values.astype(float)


media = np.mean(X_array, axis=0)
desvio = np.std(X_array, axis=0)
desvio[desvio == 0] = 1  # Evita divisão por zero
X_norm = (X_array - media) / desvio



class ClassificadorKNND:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X_tr, y_tr):
        self.X_train = X_tr
        self.y_train = y_tr

    def predict(self, X_te):
        previsoes = []
        for x in X_te:
            # Distância Euclidiana do zero
            distancias = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            k_indices = np.argsort(distancias)[:self.k]
            k_vizinhos_classes = self.y_train[k_indices]
            classes_unicas, contagens = np.unique(k_vizinhos_classes, return_counts=True)
            classe_predita = classes_unicas[np.argmax(contagens)]
            previsoes.append(classe_predita)
        return np.array(previsoes)



k_folds = 5
np.random.seed(42)
indices = np.arange(len(X_norm))
np.random.shuffle(indices)

# Divide os índices em 5 partes iguais
folds = np.array_split(indices, k_folds)

lista_acuracias = []
lista_recalls = []
lista_f1 = []

for i in range(k_folds):

    idx_teste = folds[i]
    idx_treino = np.hstack([folds[j] for j in range(k_folds) if j != i])

    X_treino, X_teste = X_norm[idx_treino], X_norm[idx_teste]
    y_treino, y_teste = y[idx_treino], y[idx_teste]


    knn = ClassificadorKNND(k=5)
    knn.fit(X_treino, y_treino)
    y_pred = knn.predict(X_teste)


    VP = np.sum((y_pred == 1) & (y_teste == 1))
    VN = np.sum((y_pred == 0) & (y_teste == 0))
    FP = np.sum((y_pred == 1) & (y_teste == 0))
    FN = np.sum((y_pred == 0) & (y_teste == 1))

    acuracia = (VP + VN) / len(y_teste) if len(y_teste) > 0 else 0
    sensibilidade = VP / (VP + FN) if (VP + FN) > 0 else 0
    precisao = VP / (VP + FP) if (VP + FP) > 0 else 0
    f1_score = (2 * precisao * sensibilidade) / (precisao + sensibilidade) if (precisao + sensibilidade) > 0 else 0

    lista_acuracias.append(acuracia)
    lista_recalls.append(sensibilidade)
    lista_f1.append(f1_score)


print(f"Amostras totais processadas: {len(X_norm)}")
print("--- Resultados Finais (Média e Desvio Padrão) ---")
print(f"Acurácia: {np.mean(lista_acuracias):.4f} (+/- {np.std(lista_acuracias):.4f})")
print(f"Sensibilidade (Recall): {np.mean(lista_recalls):.4f} (+/- {np.std(lista_recalls):.4f})")
print(f"F1-Score: {np.mean(lista_f1):.4f} (+/- {np.std(lista_f1):.4f})")

novo_paciente_bruto = [
    48, 80, 1.020, 1, 0, 'normal', 'normal', 'notpresent', 'notpresent', 121,
    36, 1.2, 135, 4.5, 15.4, 44, 7800, 5.2, 'yes', 'yes',
    'no', 'good', 'no', 'no'
]

# Converte para DataFrame temporário para aplicar as mesmas regras
df_novo = pd.DataFrame([novo_paciente_bruto], columns=X.columns)


for col in df_novo.columns:
    df_novo[col] = df_novo[col].astype(str).str.strip().str.replace('\t', '')
    df_novo[col] = df_novo[col].replace(['?', 'nan', 'NaN', ''], np.nan)

    temp_num = pd.to_numeric(df_novo[col], errors='coerce')

    if pd.api.types.is_numeric_dtype(X[col]):
        df_novo[col] = temp_num.fillna(X[col].median())  # Usa a mediana do treino
    else:
        moda = X[col].mode()[0] if not X[col].mode().empty else 0
        df_novo[col] = df_novo[col].fillna(moda)
        # Mapeia para o mesmo padrão numérico do treino
        unicos_treino = list(X[col].unique())
        val = df_novo[col].iloc[0]
        df_novo[col] = unicos_treino.index(val) if val in unicos_treino else 0

# Converte para array
x_novo_array = df_novo.values.astype(float)


x_novo_norm = (x_novo_array - media) / desvio


predicao_novo = knn.predict(x_novo_norm)


resultado_texto = "Com Doença Renal (CKD)" if predicao_novo[0] == 1 else "Saudável (Not CKD)"
print("\n--- Teste com um Dado Novo ---")
print(f"O modelo classificou o novo paciente como: **{resultado_texto}**")