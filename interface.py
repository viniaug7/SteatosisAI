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


# st.title("Trabalho PAI")
mainTab, verROITab = st.tabs(["Imagem & Cortar ROI", "Ver ROIs"])
mainContainer = mainTab.container()

if "ROIsDaImagem" not in st.session_state:
    st.session_state.ROIsDaImagem = None 
if "imagemEscolhida" not in st.session_state:
    st.session_state.imagemEscolhida = None
if "imagensVariadas" not in st.session_state:
    st.session_state.imagensVariadas = []


def escolherImagem(imagem, n,m):
    if not isinstance(imagem, np.ndarray):
        imagem = np.array(imagem)
    st.session_state.imagemEscolhida = (imagem, n, m)

def transformar_imagens_mat_em_botoes_na_sidebar(arquivo_mat):
    # Carregar arquivo .mat
    data = scipy.io.loadmat(arquivo_mat)
    data_array = data["data"]
    images = data_array["images"]
    for n in range(55):
        with st.sidebar.expander(f"Paciente {n}"):
            for m in range(10):
                imagem = images[0][n][m]
                btn = st.button(f"Imagem {m} do Paciente {n}", key=f"botao_{n}_{m}", use_container_width=True, on_click=escolherImagem, args=[imagem, n,m]) 

def carregar_arquivo_mat(caminho_arquivo):
    try:
        transformar_imagens_mat_em_botoes_na_sidebar(caminho_arquivo)
    except FileNotFoundError:
        print(f"Arquivo {caminho_arquivo} não encontrado.")
    except Exception as e:
        print(f"Erro ao carregar o arquivo: {e}")

if "arquivo_mat_carregado" not in st.session_state:
    # Carregar o arquivo .mat apenas uma vez
    carregar_arquivo_mat("./base/dataset_liver_bmodes_steatosis_assessment_IJCARS.mat")
    st.session_state.arquivo_mat_carregado = True  # Marcar como carregado

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
            

def histograma(imagem):
    image_array = np.array(imagem)
    plt.hist(image_array.flatten(), bins=256, range=(0, 256), color='black')
    plt.title('Histograma em Escala de Cinza')
    plt.xlabel('Intensidade de pixel')
    plt.ylabel('Número de pixels')
    st.pyplot(plt)
    plt.clf()


with st.sidebar.expander('Imagens diversas'):
    for nome, img in st.session_state.imagensVariadas:
        st.button(f"Imagem {nome}", key=f"botao_{randint(0, 9999999)}", on_click=escolherImagem, args=[img, None, None])

with mainContainer:
    c1, c2 = st.columns(2);
    col1 = c1.container();
    col2 = c2.container();
    if (st.session_state.imagemEscolhida is not None):
        img = st.session_state.imagemEscolhida[0]
        n = st.session_state.imagemEscolhida[1]
        m = st.session_state.imagemEscolhida[2]
        with col1:
            cropped_img = st_cropper(Image.fromarray(img), realtime_update=True, box_color='#90ee90', aspect_ratio=(20,20))
            st.session_state.ROIsDaImagem = cropped_img;
            insideC1, insideC2 = st.columns(2)
            with insideC1:
                st.button('Salvar ROI')
            with insideC2:
                if (n is not None):
                    st.write(f"ROI Paciente {n} Imagem {m}: {st.session_state.ROIsDaImagem.size}")
            histograma(img)
    if (st.session_state.ROIsDaImagem is not None):
        with col2:
            st.image(st.session_state.ROIsDaImagem, use_column_width=True)
            histograma(st.session_state.ROIsDaImagem)