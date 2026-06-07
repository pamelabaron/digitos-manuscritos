from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

pasta_dados = Path('dados')
pasta_modelos = Path('modelos')
pasta_modelos.mkdir(exist_ok=True)

arquivo_treino = pasta_dados / 'train.csv'
if not arquivo_treino.exists():
    raise FileNotFoundError('arquivo dados/train.csv nao encontrado. rode baixar_dados.py primeiro.')

dados = pd.read_csv(arquivo_treino)
x = dados.drop(columns=['label'])
y = dados['label']

x_treino, x_validacao, y_treino, y_validacao = train_test_split(
    x, y, test_size=0.15, random_state=42, stratify=y
)

modelo = Pipeline([
    ('padronizador', StandardScaler()),
    ('classificador', MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size=128,
        learning_rate_init=0.001,
        max_iter=30,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=5,
        random_state=42,
        verbose=True
    ))
])

modelo.fit(x_treino, y_treino)
previsoes = modelo.predict(x_validacao)
acuracia = accuracy_score(y_validacao, previsoes)
relatorio = classification_report(y_validacao, previsoes)

joblib.dump(modelo, pasta_modelos / 'modelo_mnist_mlp.joblib')

with open(pasta_modelos / 'metricas.txt', 'w', encoding='utf-8') as arquivo:
    arquivo.write(f'acuracia_validacao: {acuracia:.6f}\n\n')
    arquivo.write(relatorio)

print(f'acuracia_validacao: {acuracia:.6f}')
print('modelo salvo em modelos/modelo_mnist_mlp.joblib')