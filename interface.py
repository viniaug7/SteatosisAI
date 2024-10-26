# Como rodar:
# pip install streamlit, streamlit-cropper scipy, numpy, matplotlib, opencv-python
# streamlit run interface.py
import streamlit as st
from random import randint
import scipy.io
from streamlit_cropper import st_cropper
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from streamlit_image_zoom import image_zoom
import glcm as gl
import cv2



st.set_page_config(layout="wide")


# st.title("Trabalho PAI")
mainTab, verROITab = st.tabs(["Imagem & Cortar ROI", "Ver ROIs"])
mainContainer = mainTab.container()

if "ROIsSalvos" not in st.session_state:
    st.session_state.ROIsSalvos = []
if "ROIDaImagem" not in st.session_state:
    st.session_state.ROIDaImagem = None 
if "imagemEscolhida" not in st.session_state:
    st.session_state.imagemEscolhida = None
if "imagensVariadas" not in st.session_state:
    st.session_state.imagensVariadas = []

def gerar_id_unico(offset=0):
    if 'ultimo_id' not in st.session_state:
        st.session_state['ultimo_id'] = 0
    st.session_state['ultimo_id'] += 1
    return st.session_state['ultimo_id'] + offset

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


# Upload de arquivo no topo do sidebar
arquivo = st.sidebar.file_uploader("", type=["mat", "jpeg", "jpg", "png"])
    
carregar_arquivo_mat("./base/dataset_liver_bmodes_steatosis_assessment_IJCARS.mat")

# Se o arquivo existir e o nome dele ja nao estiver na lista de imagens variadas
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
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            imagem = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    gl.glcm(imagem, [1,2,4,8])

with st.sidebar.expander('Imagens diversas'):
    for nome, img in st.session_state.imagensVariadas:
        st.button(f"Imagem {nome}", key=f"botao_{randint(0, 9999999)}", on_click=escolherImagem, args=[img, None, None])

with mainContainer:
    c1, c2 = st.columns(2);
    c3, c4 = st.columns(2)
    col1 = c1.container();
    col2 = c2.container();
    if (st.session_state.imagemEscolhida is not None):
        img = st.session_state.imagemEscolhida[0]
        n = st.session_state.imagemEscolhida[1]
        m = st.session_state.imagemEscolhida[2]
        with col1:
            cropped_img = st_cropper(Image.fromarray(img), realtime_update=True, box_color='#90ee90', aspect_ratio=(20,20))
            st.session_state.ROIDaImagem = cropped_img;
            insideC1, insideC2 = st.columns(2)
            with insideC1:
                if (st.button('Salvar ROI')):
                    # Salvar a ROI em session_state, guardando n e m
                    st.session_state.ROIsSalvos.append((st.session_state.ROIDaImagem, n, m, gerar_id_unico()))
            with insideC2:
                if (n is not None):
                    st.write(f"ROI Paciente {n} Imagem {m}: {st.session_state.ROIDaImagem.size}")
        with c3:
            histograma(img)
    if (st.session_state.ROIDaImagem is not None):
        with col2:
            st.image(st.session_state.ROIDaImagem, use_column_width=True)
        with c4:
            histograma(st.session_state.ROIDaImagem)
        

def calcular_hi(roi_figado, roi_rim):
    media_figado = np.mean(roi_figado)
    media_rim = np.mean(roi_rim)
    hi = media_figado / media_rim
    return hi

def normalizar_figado(roi_figado, hi):
    # multpalica os valores de pixel pelo HI e dpois arredonda
    roi_normalizada = np.clip(np.round(roi_figado * hi), 0, 255).astype(np.uint8)
    return roi_normalizada


with verROITab:
    mainc1, mainc2, mainc3 = st.columns(3)
    i = 0;
    # selecione figado entre ROIsSalvos
    figado = mainc1.selectbox("Selecione o figado", [f"{id}" for roi, n, m, id in st.session_state.ROIsSalvos], key="figado")
    # selecionar rim
    rim = mainc2.selectbox("Selecione o rim", [f"{id}" for roi, n, m, id in st.session_state.ROIsSalvos], key="rim")

    if (mainc1.button("Processar")):
        # achar ROIsalvo por ID
        figadoROI = [(roi, n,m, id) for roi, n, m, id in st.session_state.ROIsSalvos if str(id) == figado][0]
        rimROI = [(roi, n,m, id) for roi, n, m, id in st.session_state.ROIsSalvos if str(id) == rim][0]
        hi = calcular_hi(figadoROI[0], rimROI[0])
        roi_figado_normalizada = normalizar_figado(figadoROI[0], hi)
        st.session_state.ROIsSalvos.append((roi_figado_normalizada, figadoROI[1], figadoROI[2], gerar_id_unico(1000)))


    c1,c2,c3 = st.columns(3);
    for roi, n, m, id in st.session_state.ROIsSalvos:
        colatual = c1 if i % 3 == 0 else c2 if i % 3 == 1 else c3
        i += 1
        with colatual:
            st.write(f"ROI Paciente {n} Imagem {m} ID {id}: {roi.size}")
            image_zoom(roi, size=(420, 420))
            # st.image(roi, use_column_width=True)
            histograma(roi)