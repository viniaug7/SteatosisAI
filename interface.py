# Como rodar:
# pip install streamlit, streamlit-cropper scipy, numpy, matplotlib
# streamlit run interface.py
import streamlit as st
from random import randint
import scipy.io
from streamlit_cropper import st_cropper
import numpy as np
import io
import matplotlib.pyplot as plt
from PIL import Image


st.set_page_config(layout="wide")


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
mainCol1, mainCol2 = container.columns(2)

if "imagemEscolhida" not in st.session_state:
    st.session_state.imagemEscolhida = []
if "imagensVariadas" not in st.session_state:
    st.session_state.imagensVariadas = []


def escolherImagem(imagem, n,m):
    # mainCol1.image(imagem, caption="Imagem escolhida")
    st.session_state.imagemEscolhida = imagem

def transformar_imagens_mat_em_botoes_na_sidebar(arquivo_mat):
    # Carregar arquivo .mat
    data = scipy.io.loadmat(arquivo_mat)
    data_array = data["data"]
    images = data_array["images"]
    for n in range(55):
        with st.sidebar.expander(f"Paciente {n+1}"):
            for m in range(10):
                imagem = images[0][n][m]
                btn = st.button(f"Imagem {m+1} do Paciente {n+1}", key=f"botao_{n}_{m}", use_container_width=True, on_click=escolherImagem, args=[imagem, n,m]) 




# Upload de arquivo no topo do sidebar
st.sidebar.title("Adicionar Foto")
arquivo = st.sidebar.file_uploader("Carregar uma imagem", type=["mat", "jpeg", "jpg", "png"])

if arquivo and arquivo.name not in [nome for nome, _  in st.session_state.imagensVariadas]:
    imagem= None
    # Verifica o tipo de arquivo
    if arquivo.type in ["image/jpeg", "image/png"]:
        imagem = Image.open(arquivo)
        st.session_state.imagensVariadas.append((arquivo.name, imagem))

    elif arquivo.type == "application/octet-stream":  # Para arquivos .mat
        transformar_imagens_mat_em_botoes_na_sidebar(arquivo)
            


with st.sidebar.expander('Imagens diversas'):
    for nome, img in st.session_state.imagensVariadas:
        st.button(f"Imagem {nome}", key=f"botao_{randint(0, 9999999)}", on_click=escolherImagem, args=[img, None, None])

with mainCol1:
    if (len(st.session_state.imagemEscolhida) != 0):
        cropped_img = st_cropper(Image.fromarray(st.session_state.imagemEscolhida), realtime_update=False, box_color='#0000FF', aspect_ratio=(1,1))

with mainCol2:
    if (len(st.session_state.imagemEscolhida) != 0):
        image_array = np.array(st.session_state.imagemEscolhida)
        plt.hist(image_array.flatten(), bins=256, range=(0, 256), color='black')
        plt.title('Histograma em Escala de Cinza')
        plt.xlabel('Intensidade de pixel')
        plt.ylabel('Número de pixels')
        mainCol2.pyplot(plt)
        plt.clf()