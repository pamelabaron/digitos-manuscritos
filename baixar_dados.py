from pathlib import Path
import subprocess
import sys

pasta_dados = Path('dados')
pasta_dados.mkdir(exist_ok=True)

comando = [
    'kaggle', 'competitions', 'download',
    '-c', 'digit-recognizer',
    '-p', str(pasta_dados)
]

try:
    subprocess.run(comando, check=True)
except FileNotFoundError:
    print('kaggle cli nao encontrado. instale com: pip install kaggle')
    sys.exit(1)
except subprocess.CalledProcessError:
    print('falha ao baixar dados. confira se a api do kaggle esta configurada.')
    sys.exit(1)

zip_arquivo = pasta_dados / 'digit-recognizer.zip'
if zip_arquivo.exists():
    import zipfile
    with zipfile.ZipFile(zip_arquivo, 'r') as zip_ref:
        zip_ref.extractall(pasta_dados)
    print('download e extracao concluidos em', pasta_dados)
else:
    print('arquivo zip nao encontrado, mas o comando foi executado.')