# Como rodar:
# pip install streamlit, scipy, numpy
# streamlit run interface.py
import streamlit as st
import scipy.io
import numpy as np
import io
from PIL import Image




# do ipynb do canvas:
## carregar dados
#data = scipy.io.loadmat(path_data)
# é um dicionário, 'data' contém as imagens
# data.keys()
# accessar as imagen de 'data'
#data_array = data['data']
#images = data_array['images']
# Acesso à imagem m do paciente n
# n varia de 0 a 54, m de 0 a 9
#n=1
#m=5
#imagem = images[0][n][m]
#print(imagem.shape)
## plotar a imagem
#plt.figure(figsize=(9,9))
#plt.imshow(first_image, cmap='gray')  # Use 'gray' for grayscale images
#plt.axis('off')  # Hide axes for better visualization
#plt.show()


st.title("Trabalho PAI")
container = st.container()

def imagemEscolhida(imagem, n,m):
    container.image(imagem, caption="Imagem escolhida")

def transformar_imagens_mat_em_botoes_na_sidebar(arquivo_mat):
    # Carregar arquivo .mat
    data = scipy.io.loadmat(arquivo_mat)
    data_array = data["data"]
    images = data_array["images"]
    for n in range(55):
        with st.sidebar.expander(f"Paciente {n+1}"):
            for m in range(10):
                imagem = images[0][n][m]
                btn = st.button(f"Imagem {m+1} do Paciente {n+1}", key=f"botao_{n}_{m}", use_container_width=True, on_click=imagemEscolhida, args=[imagem, n,m]) 




# Upload de arquivo no topo do sidebar
st.sidebar.title("Adicionar Foto")
arquivo = st.sidebar.file_uploader("Carregar uma imagem", type=["mat", "jpeg", "jpg", "png"])

#aqui é onde vamos receber as imagens (esta no sidebar)
if arquivo:
    imagem= None
    # Verifica o tipo de arquivo
    if arquivo.type in ["image/jpeg", "image/png"]:
        imagem = Image.open(arquivo)
        sdContainer = st.sidebar.container()
        with sdContainer.expander('Imagens diversas'):
            btn = st.button(f"Imagem {arquivo.name}", key=f"botao_{arquivo.name}", on_click=imagemEscolhida, args=[imagem, None, None]) 

    elif arquivo.type == "application/octet-stream":  # Para arquivos .mat
        transformar_imagens_mat_em_botoes_na_sidebar(arquivo)
            

