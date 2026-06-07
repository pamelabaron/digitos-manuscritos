from pathlib import Path
import tkinter as tk
from tkinter import messagebox
import joblib
import numpy as np
from PIL import Image, ImageDraw, ImageOps, ImageTk

largura_canvas = 280
altura_canvas = 280
cor_fundo = 'black'
cor_traco = 'white'
espessura = 18

class AplicacaoDigitos:
    def __init__(self, janela):
        self.janela = janela
        self.janela.title('classificacao de digitos manuscritos')
        self.janela.resizable(False, False)

        caminho_modelo = Path('modelos') / 'modelo_mnist_mlp.joblib'
        if not caminho_modelo.exists():
            messagebox.showerror('erro', 'modelo nao encontrado. rode treinar_modelo.py antes.')
            self.janela.destroy()
            return

        self.modelo = joblib.load(caminho_modelo)
        self.imagem = Image.new('L', (largura_canvas, altura_canvas), color=0)
        self.desenho = ImageDraw.Draw(self.imagem)
        self.ultimo_x = None
        self.ultimo_y = None

        frame_principal = tk.Frame(self.janela, padx=12, pady=12)
        frame_principal.pack()

        self.canvas = tk.Canvas(
            frame_principal,
            width=largura_canvas,
            height=altura_canvas,
            bg=cor_fundo,
            highlightthickness=1,
            highlightbackground='#888'
        )
        self.canvas.grid(row=0, column=0, columnspan=3)

        self.canvas.bind('<Button-1>', self.iniciar_desenho)
        self.canvas.bind('<B1-Motion>', self.desenhar)
        self.canvas.bind('<ButtonRelease-1>', self.parar_desenho)

        self.rotulo_resultado = tk.Label(
            frame_principal,
            text='desenhe um digito de 0 a 9 e clique em classificar',
            font=('arial', 12),
            pady=10
        )
        self.rotulo_resultado.grid(row=1, column=0, columnspan=3)

        botao_classificar = tk.Button(frame_principal, text='classificar', width=16, command=self.classificar)
        botao_limpar = tk.Button(frame_principal, text='limpar', width=16, command=self.limpar)
        botao_sair = tk.Button(frame_principal, text='sair', width=16, command=self.janela.destroy)

        botao_classificar.grid(row=2, column=0, padx=4, pady=4)
        botao_limpar.grid(row=2, column=1, padx=4, pady=4)
        botao_sair.grid(row=2, column=2, padx=4, pady=4)

        self.rotulo_preview = tk.Label(frame_principal)
        self.rotulo_preview.grid(row=3, column=0, columnspan=3, pady=8)

    def iniciar_desenho(self, evento):
        self.ultimo_x = evento.x
        self.ultimo_y = evento.y

    def desenhar(self, evento):
        x1, y1 = self.ultimo_x, self.ultimo_y
        x2, y2 = evento.x, evento.y
        self.canvas.create_line(x1, y1, x2, y2, fill=cor_traco, width=espessura, capstyle=tk.ROUND, smooth=True)
        self.desenho.line((x1, y1, x2, y2), fill=255, width=espessura)
        self.ultimo_x = x2
        self.ultimo_y = y2

    def parar_desenho(self, _evento):
        self.ultimo_x = None
        self.ultimo_y = None

    def limpar(self):
        self.canvas.delete('all')
        self.imagem = Image.new('L', (largura_canvas, altura_canvas), color=0)
        self.desenho = ImageDraw.Draw(self.imagem)
        self.rotulo_resultado.config(text='desenhe um digito de 0 a 9 e clique em classificar')
        self.rotulo_preview.config(image='')
        self.rotulo_preview.image = None

    def preprocessar_imagem(self):
        imagem_invertida = ImageOps.invert(self.imagem)
        caixa = imagem_invertida.getbbox()
        if caixa is None:
            return None, None

        digito = self.imagem.crop(caixa)
        largura, altura = digito.size
        lado = max(largura, altura) + 20

        tela_quadrada = Image.new('L', (lado, lado), color=0)
        posicao = ((lado - largura) // 2, (lado - altura) // 2)
        tela_quadrada.paste(digito, posicao)

        imagem_28 = tela_quadrada.resize((20, 20), Image.Resampling.LANCZOS)
        fundo_28 = Image.new('L', (28, 28), color=0)
        fundo_28.paste(imagem_28, (4, 4))

        vetor = np.array(fundo_28, dtype=np.float32).reshape(1, -1)
        return vetor, fundo_28

    def classificar(self):
        vetor, preview = self.preprocessar_imagem()
        if vetor is None:
            messagebox.showwarning('aviso', 'desenhe um digito antes de classificar.')
            return

        probabilidades = self.modelo.predict_proba(vetor)[0]
        classe = int(np.argmax(probabilidades))
        confianca = float(np.max(probabilidades)) * 100
        self.rotulo_resultado.config(text=f'resultado previsto: {classe} | confianca: {confianca:.2f}%')

        imagem_preview = preview.resize((112, 112), Image.Resampling.NEAREST)
        imagem_preview = ImageTk.PhotoImage(imagem_preview)
        self.rotulo_preview.config(image=imagem_preview)
        self.rotulo_preview.image = imagem_preview

if __name__ == '__main__':
    janela = tk.Tk()
    AplicacaoDigitos(janela)
    janela.mainloop()