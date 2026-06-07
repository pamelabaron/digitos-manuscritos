# classificacao de digitos manuscritos com python

Projeto em Python para treinar um classificador de dígitos manuscritos com base no desafio **Digit Recognizer** do Kaggle e no exemplo de MLP do notebook referenciado. O modelo usa `MLPClassifier` do scikit-learn e a aplicação inclui uma interface interativa em `tkinter` para desenhar um dígito com o mouse e classificá-lo.

Referências utilizadas:
- Kaggle notebook: [MNIST Classification using Multilayer Perceptron](https://www.kaggle.com/code/jonathankristanto/mnist-classification-using-multilayer-perceptron)
- Kaggle competição: [Digit Recognizer](https://www.kaggle.com/c/digit-recognizer/data)
- Exemplo visual de interação: [Digits Recognition MLP](https://trekhleb.dev/machine-learning-experiments/#/experiments/DigitsRecognitionMLP)

---
Pâmela Baron e DerecK Conink
---

## estrutura do projeto

```text
├── README.md
├── app_interativa.py
├── baixar_dados.py
├── treinar_modelo.py
├── prever_csv_teste.py
├── requirements.txt
├── .gitignore
├── resultados/
│   ├── metricas.txt
│   └── submission.csv

```

## requisitos

- Windows com Python
- VSCode com extensão Python instalada
- Conta no Kaggle
- API do Kaggle configurada no Windows


## como configurar o kaggle no windows

1. Entre em [https://www.kaggle.com/settings](https://www.kaggle.com/settings).
2. Em **API**, gere um novo token.
3. Baixe o arquivo `kaggle.json`.
4. Coloque esse arquivo em:

```text
C:\Users\usuario\.kaggle\kaggle.json
```

## como rodar

### 1. clonar

```bash
git clone <url-do-repositorio>
cd digitos manuscritos
```

### 2. criar ambiente virtual

No terminal do VSCode, execute:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. instalar dependências

```bash
pip install -r requirements.txt
```

### 4. baixar os dados do kaggle

```bash
python baixar_dados.py
```

Vai gerar os arquivos `train.csv`, `test.csv` e `sample_submission.csv` dentro da pasta `dados`.

### 5. treinar o modelo

```bash
python treinar_modelo.py
```

Por fim o arquivo `modelos/modelo_mnist_mlp.joblib` será criado.

### 6. abrir a interface interativa

```bash
python app_interativa.py
```

Na janela aberta:
- desenhe um dígito com o mouse
- clique em `classificar`
- veja o dígito previsto e a confiança do modelo
- use `limpar` para desenhar novamente

### 7. gerar arquivo de submissão para o kaggle

```bash
python prever_csv_teste.py
```

Esse comando cria o arquivo `submission.csv`.

## observações sobre a implementação

- O modelo utiliza `StandardScaler` + `MLPClassifier` em pipeline.
- A base de treino do Kaggle possui a coluna `label` e 784 colunas de pixels, representando imagens de 28x28.
- A interface desenha em resolução maior e depois faz o pré-processamento para 28x28 antes da inferência.
- O recorte automático remove áreas vazias para melhorar a centralização do dígito.

