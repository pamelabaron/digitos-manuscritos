from pathlib import Path
import joblib
import pandas as pd

pasta_dados = Path('dados')
pasta_modelos = Path('modelos')

arquivo_teste = pasta_dados / 'test.csv'
arquivo_modelo = pasta_modelos / 'modelo_mnist_mlp.joblib'

if not arquivo_teste.exists():
    raise FileNotFoundError('arquivo dados/test.csv nao encontrado.')
if not arquivo_modelo.exists():
    raise FileNotFoundError('modelo nao encontrado. rode treinar_modelo.py primeiro.')

modelo = joblib.load(arquivo_modelo)
dados_teste = pd.read_csv(arquivo_teste)
previsoes = modelo.predict(dados_teste)

submissao = pd.DataFrame({
    'ImageId': range(1, len(previsoes) + 1),
    'Label': previsoes
})

submissao.to_csv('submission.csv', index=False)
print('arquivo submission.csv gerado com sucesso.')