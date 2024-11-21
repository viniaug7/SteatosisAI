#NT:(2322434)mod4 = 2  ->  Momentos invariantes de Hu, se NT=2
#Eric Miranda: 771803
#Gustavo Lorenzo: 782633
#Vinicius Augusto: 767998
# Como rodar:
# pip install streamlit streamlit-cropper scipy numpy matplotlib setuptools opencv-python streamlit_image_zoom scikit-image 
# streamlit run interface.py
import streamlit as st
from random import randint
import scipy.io
from streamlit_cropper import st_cropper
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from streamlit_image_zoom import image_zoom
from skimage.feature import graycomatrix, graycoprops
import pandas as pd
import cv2
import os
#from scipy.stats import entropy opcional, escolher qual processa mais rapido a entropia -> fiz na mão

SIMPLE_CSV_FILENAME = 'dados_das_rois.csv'
CLASSES_CSV_FILENAME = 'rois_com_classes.csv'

if not os.path.isfile(SIMPLE_CSV_FILENAME):
    with open(SIMPLE_CSV_FILENAME, 'w') as f:
        f.write('n,m\n')  
if not os.path.isfile(CLASSES_CSV_FILENAME):
    with open(CLASSES_CSV_FILENAME, 'w') as f:
        f.write('nome\n') 


def salvarROIEmCSVePasta(imagemFigado, n, m, coordsFigado, coordsRim, HI, dadosGLCM): 
    nFormatado = str(n).zfill(2)
    mFormatado = str(m)
    nomeDoArquivo = f'ROI_{nFormatado}_{mFormatado}.jpg'
    imagemFigado.save(nomeDoArquivo)

    coordsFigadoASalvar = f"{coordsFigado['left']}-{coordsFigado['top']}"
    coordsRimASalvar = f"{coordsRim['left']}-{coordsRim['top']}"
    
    classe = 'saudavel' if n <= 16 else 'esteatose'

    if os.path.isfile(SIMPLE_CSV_FILENAME):
        roi_data = pd.read_csv(SIMPLE_CSV_FILENAME)
    else:
        roi_data = pd.DataFrame(columns=['nome_arquivo', 'classe', 'coords_figado', 'coords_rim', 'HI'])

    ja_existe = roi_data[(roi_data['n'] == n) & (roi_data['m'] == m)]
    
    if not ja_existe.empty:
        roi_data.loc[ja_existe.index, 'coords_figado'] = coordsFigadoASalvar
        roi_data.loc[ja_existe.index, 'coords_rim'] = coordsRimASalvar
        roi_data.loc[ja_existe.index, 'HI'] = HI
    else:
        nova = pd.DataFrame({
            'nome_arquivo': [nomeDoArquivo],
            'classe': [classe],
            'coords_figado': [coordsFigadoASalvar],
            'coords_rim': [coordsRimASalvar],
            'HI': [HI],
            'n': [n],
            'm': [m]
        })
        roi_data = pd.concat([roi_data, nova], ignore_index=True)

    roi_data.to_csv(SIMPLE_CSV_FILENAME, index=False)

    data_para_csv = {
        'nome': [nomeDoArquivo],
        'classe': [classe],
    }

    distancias = [1, 2, 4, 8]
    i = 0
    for result in dadosGLCM:
        data_para_csv[f'homogeonidade_{distancias[i]}'] = [result['homogeneity']]
        data_para_csv[f'distancia_{distancias[i]}'] = [result['distance']]
        data_para_csv[f'entropia_{distancias[i]}'] = [result['entropy']]
        data_para_csv[f'momentos_hu_{distancias[i]}'] = [result['hu_moments']]
        i += 1

    if os.path.isfile(CLASSES_CSV_FILENAME):
        classes_data = pd.read_csv(CLASSES_CSV_FILENAME)
    else:
        classes_data = pd.DataFrame(columns=['nome', 'classe'])

    classes_data = pd.concat([classes_data, pd.DataFrame(data_para_csv)], ignore_index=True)

    classes_data.to_csv(CLASSES_CSV_FILENAME, index=False)


st.set_page_config(layout="wide")


# st.title("Trabalho PAI")
mainTab, verROITab, classificarTab  = st.tabs(["Imagem & Cortar ROI", "Ver ROIs", "Classificar"])
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

@st.cache_data
def carregar_imagem(upload):
    return Image.open(upload)

@st.cache_data
def loadMat(arquivo_mat):
    return scipy.io.loadmat(arquivo_mat)


def transformar_imagens_mat_em_botoes_na_sidebar(arquivo_mat):
    # Carregar arquivo .mat
    data = loadMat(arquivo_mat)
    data_array = data["data"]
    images = data_array["images"]
    for n in range(55):
        with st.sidebar.expander(f"Paciente {n}"):
            for m in range(10):
                imagem = images[0][n][m]
                st.button(f"Imagem {m} do Paciente {n}", key=f"botao_{n}_{m}", use_container_width=True, on_click=escolherImagem, args=[imagem, n,m]) 


def carregar_arquivo_mat(caminho_arquivo):
    try:
        transformar_imagens_mat_em_botoes_na_sidebar(caminho_arquivo)
    except FileNotFoundError:
        print('Arquivo pre esperado nao encontrado, voce pode carregar ele com o browse na parte suprior esquerda')
    except Exception as e:
        st.error(f"Erro ao carregar o arquivo: {e}")


# Upload de arquivo no topo do sidebar

arquivo = st.sidebar.file_uploader("", type=["mat", "jpeg", "jpg", "png"])

   
carregar_arquivo_mat("./base/dataset_liver_bmodes_steatosis_assessment_IJCARS.mat")

# Se o arquivo existir e o nome dele ja nao estiver na lista de imagens variadas
if arquivo and arquivo.name not in [nome for nome, _  in st.session_state.imagensVariadas]:
    imagem= None
    # Verifica o tipo de arquivo
    if arquivo.type in ["image/jpeg", "image/png"]:
        imagem = carregar_imagem(arquivo)
        st.session_state.imagensVariadas.append((arquivo.name, imagem))

    elif arquivo.type == "application/octet-stream":  # Para arquivos .mat
        transformar_imagens_mat_em_botoes_na_sidebar(arquivo)


@st.cache_data
def circular_offsets(distance):
    angles = []
    if distance == 1:
         angles = [-2.356194490192345, 3.141592653589793, 2.356194490192345, -1.5707963267948966, 1.5707963267948966, -0.7853981633974483, 0.0, 0.7853981633974483]
    elif distance == 2:
         angles = [-2.677945044588987, 3.141592653589793, 2.677945044588987, -2.0344439357957027, 2.0344439357957027, -1.5707963267948966, 1.5707963267948966, -1.1071487177940904, 1.1071487177940904, -0.4636476090008061, 0.0, 0.4636476090008061]
    elif distance == 4:
         angles = [-2.677945044588987, -2.896613990462929, 3.141592653589793, 2.896613990462929, 2.677945044588987, -2.356194490192345, -2.5535900500422257, 2.5535900500422257, 2.356194490192345, -2.0344439357957027, -2.158798930342464, 2.158798930342464, 2.0344439357957027, -1.8157749899217608, 1.8157749899217608, -1.5707963267948966, 1.5707963267948966, -1.3258176636680326, 1.3258176636680326, -1.1071487177940904, -0.982793723247329, 0.982793723247329, 1.1071487177940904, -0.7853981633974483, -0.5880026035475675, 0.5880026035475675, 0.7853981633974483, -0.4636476090008061, -0.24497866312686414, 0.0, 0.24497866312686414, 0.4636476090008061]
    elif distance == 8:
         angles = [-2.896613990462929, -3.017237659043032, 3.141592653589793, 3.017237659043032, 2.896613990462929, -2.62244653934327, -2.7367008673047097, 2.7367008673047097, 2.62244653934327, -2.356194490192345, -2.44685437739309, 2.44685437739309, 2.356194490192345, -2.2655346029916, 2.2655346029916, -2.0899424410414196, 2.0899424410414196, -1.97568811307998, 1.97568811307998, -1.8157749899217608, 1.8157749899217608, -1.695151321341658, 1.695151321341658, -1.5707963267948966, 1.5707963267948966, -1.446441332248135, 1.446441332248135, -1.3258176636680326, 1.3258176636680326, -1.1659045405098132, 1.1659045405098132, -1.0516502125483738, 1.0516502125483738, -0.8760580505981934, 0.8760580505981934, -0.7853981633974483, -0.6947382761967031, 0.6947382761967031, 0.7853981633974483, -0.5191461142465229, -0.40489178628508343, 0.40489178628508343, 0.5191461142465229, -0.24497866312686414, -0.12435499454676144, 0.0, 0.12435499454676144, 0.24497866312686414]
    else: 
         st.error("Distancia nao definida")
    # Como as distancias sao constantes, salvei os angulos para evitar processamento desnecessario(tempo)
    # Os angulos foram calculados da seguinte maneira:
    #for x in range(-distance, distance + 1):
    #    for y in range(-distance, distance + 1):
    #        if np.round(np.sqrt(x**2 + y**2)) == distance and (x, y) != (0, 0):
    #            angles.append(np.arctan2(y, x))  # Mudado para formato radiano
    return angles

@st.cache_data
def glcm(image, distances=[1,2,4,8], gray_levels=256, symmetric=True, normed=True):
    results = []  # Lista para armazenar resultados
    for distance in distances:
        image_copy = np.copy(image)

        # Obter deslocamentos circulares para a distância atual
        angles = circular_offsets(distance)
        
        # GLCM
        glcm = graycomatrix(image_copy, [distance], angles, gray_levels, symmetric, normed)

        # Homogeneidade
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
        
        # Entropia
        glcm_probabilities = glcm / glcm.sum()
        glcm_probabilities_flat = glcm_probabilities.flatten()
        glcm_probabilities_flat = glcm_probabilities_flat[glcm_probabilities_flat > 0]
        glcm_entropy = -np.sum(glcm_probabilities_flat * np.log2(glcm_probabilities_flat))

        # A cv2.moment aceita somente uma matriz de dimensão 2D, a glcm é 4D
        glcm_2d = np.sum(glcm, axis=(2, 3))
        
        # Normalizar GLCM para ser uma imagem de intensidade
        glcm_normalized = (glcm_2d / glcm_2d.max() * 255).astype(np.uint8)

        # Calcular momento invariante de Hu
        moments = cv2.moments(glcm_normalized)
        hu_moments = cv2.HuMoments(moments).flatten()

        # Armazenar resultados em um dicionário
        result = {
            'distance': distance,
            'homogeneity': homogeneity,
            'entropy': glcm_entropy,
            'hu_moments': hu_moments.tolist()  # Converter para lista para facilitar o retorno
        }
        results.append(result)  # Adicionar resultado à lista

    return results  # Retornar todos os resultados

#precisa ter dois codigos de histograma para que o @st.cache_data funcione de acordo        
#e evite o processamento constante da imagem selecionada a cada iteracao
@st.cache_data
def histograma(imagem):
    image_array = np.array(imagem)
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            imagem = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    plt.hist(image_array.flatten(), bins=256, range=(0, 256), color='black')
    plt.title('Histograma em Escala de Cinza da Imagem')
    plt.xlabel('Intensidade de pixel')
    plt.ylabel('Número de pixels')
    st.pyplot(plt)
    plt.clf()
#processado somente quando salvar a roi
@st.cache_data
def histograma_roi(imagem):
    image_array = np.array(imagem)
    plt.hist(image_array.flatten(), bins=256, range=(0, 256), color='black')
    plt.title('Histograma em Escala de Cinza da ROI')
    plt.xlabel('Intensidade de pixel')
    plt.ylabel('Número de pixels')
    st.pyplot(plt)
    plt.clf()
    

with st.sidebar.expander('Imagens diversas'):
    for nome, img in st.session_state.imagensVariadas:
        st.button(f"Imagem {nome}", key=f"botao_{randint(0, 9999999)}", on_click=escolherImagem, args=[img, None, None])


def makeBox(Pil_imagem, aspect_ratio): 
    width, height = Pil_imagem.size
    # No final da 28 28, deve ser largura da box
    b_width = 27
    b_height = 27
    left = (width - b_width) // 2
    top = (height - b_height) // 2
    return { 'left': left, 'top': top, 'width': b_width, 'height': b_height }



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

            aTupla = st_cropper(Image.fromarray(img), return_type='both', stroke_width=1, key=f'cropper_{n}_{m}', box_algorithm=makeBox, realtime_update=True, box_color='#90ee90', aspect_ratio=(20,20)) 
            cropped_img  = aTupla[0]
            cropped_img_box_coords = aTupla[1]
            print(cropped_img_box_coords)
            st.session_state.ROIDaImagem = cropped_img;
            insideC1, insideC2 = st.columns(2)
            with insideC1:
                if (st.button('Salvar ROI')):
                    # Salvar a ROI em session_state, guardando n e m
                    glcmData = glcm(st.session_state.ROIDaImagem)
                    st.session_state.ROIsSalvos.append((st.session_state.ROIDaImagem, n, m, gerar_id_unico(), cropped_img_box_coords, glcmData))
                    st.success('ROI Salvo com sucesso')
            with insideC2:
                if (n is not None):
                    w = st.session_state.ROIDaImagem.size[0]
                    h = st.session_state.ROIDaImagem.size[1]
                    st.write(f"ROI Paciente {n} Imagem {m}: ({w}x{h})")
        with c3:
            histograma(img)
    if (st.session_state.ROIDaImagem is not None):
        with col2:
            st.image(st.session_state.ROIDaImagem, use_container_width=True)
        
        
def calcular_hi(roi_figado, roi_rim):
    media_figado = np.mean(roi_figado)
    media_rim = np.mean(roi_rim)
    hi = media_figado / media_rim
    return hi

def normalizar_figado(roi_figado, hi):
    # multpalica os valores de pixel pelo HI e dpois arredonda
    roi_normalizada = np.clip(np.round(roi_figado * hi), 0, 255).astype(np.uint8)
    return roi_normalizada

def tabela_glcm(dadosGLCM):
    data_for_df = []
    for result in dadosGLCM:
        data_for_df.append({
            'Distancia': result['distance'],
            'Homogeonidade': result['homogeneity'],
            'Entropia': result['entropy'],
            'Momentos HU': result['hu_moments'] 
        })

    results_df = pd.DataFrame(data_for_df)

    st.write("Resultados GLCM:")
    st.dataframe(results_df)


with verROITab:
    mainc1, mainc2, mainc3 = st.columns(3)
    i = 0;
    # selecione figado entre ROIsSalvos
    figado = mainc1.selectbox("Selecione o figado", [f"{id}" for roi, n, m, id, coords, glcmData in st.session_state.ROIsSalvos], key="figado")
    # selecionar rim
    rim = mainc2.selectbox("Selecione o rim", [f"{id}" for roi, n, m, id, coords, glcmData in st.session_state.ROIsSalvos], key="rim")

    if (mainc1.button("Normalizar figado e Salvar")):
        # achar ROIsalvo por ID
        figadoROI = [(roi, n,m, id, coords) for roi, n, m, id, coords, glcmData in st.session_state.ROIsSalvos if str(id) == figado][0]
        rimROI = [(roi, n,m, id, coords) for roi, n, m, id, coords, glcmData in st.session_state.ROIsSalvos if str(id) == rim][0]
        hi = calcular_hi(figadoROI[0], rimROI[0])
        roi_figado_normalizada = normalizar_figado(figadoROI[0], hi)
        glcmData = glcm(roi_figado_normalizada)
        st.session_state.ROIsSalvos.append((roi_figado_normalizada, figadoROI[1], figadoROI[2], gerar_id_unico(1000), figadoROI[3], glcmData))
        salvarROIEmCSVePasta(figadoROI[0], figadoROI[1], figadoROI[2], figadoROI[4], rimROI[4], hi, glcmData)
        st.success('Imagem .jpg salva nessa pasta e dados colocados nos csvs')


    c1,c2,c3 = st.columns(3);
    for roi, n, m, id, coords, glcmData in st.session_state.ROIsSalvos:
        colatual = c1 if i % 3 == 0 else c2 if i % 3 == 1 else c3
        i += 1
        with colatual:
            thiscont = st.container(key=f'{n}_{m}_{id}_{roi.size}')
            if isinstance(roi.size, int):
                w = 28
                h = 28
            else:  # Assume que roi.size é uma tupla
                w = roi.size[0]
                h = roi.size[1]
            
            with thiscont:
                st.write(f"ROI Paciente {n} Imagem {m} ID {id}: {w}x{h}")
                image_zoom(roi, size=(420, 420))
                histograma_roi(roi)
                tabela_glcm(glcmData)
            #---------------------------------------------------------------------------------------------------------
            #colocar para aparecer as variaveis: escolher como aparecer