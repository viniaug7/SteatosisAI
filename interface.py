import streamlit as st
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io

# Função para carregar arquivos .mat
def load_mat_image(file):
    mat = scipy.io.loadmat(file)
    # Supondo que a imagem esteja armazenada na chave 'image'
    if 'image' in mat:
        return mat['image']
    else:
        st.error("A chave 'image' não foi encontrada no arquivo .mat.")
        return None

# Função para carregar arquivos de imagem (PNG, JPG)
def load_image(file):
    image = Image.open(file)
    return image

# Interface Streamlit
st.title("Leitor de Imagens")

# Menu para selecionar o tipo de arquivo
file_type = st.selectbox("Escolha o tipo de arquivo", ['.mat', '.png', '.jpg'])

# Carregar o arquivo baseado no tipo selecionado
uploaded_file = st.file_uploader(f"Carregue um arquivo {file_type}", type=[file_type.strip('.')])

if uploaded_file is not None:
    if file_type == '.mat':
        # Exibe imagem de arquivo .mat
        image_data = load_mat_image(uploaded_file)
        if image_data is not None:
            st.image(image_data, caption="Imagem do arquivo .mat", use_column_width=True)
    else:
        # Exibe imagem de arquivos PNG ou JPG
        image = load_image(uploaded_file)
        st.image(image, caption=f"Imagem do arquivo {file_type}", use_column_width=True)
        image_array = np.array(image)
        st.write("Histograma para Imagem em Escala de Cinza")
        plt.hist(image_array.flatten(), bins=256, range=(0, 256), color='black')
        plt.title('Histograma em Escala de Cinza')
        plt.xlabel('Intensidade de pixel')
        plt.ylabel('Número de pixels')
        
        # Exibe o gráfico no Streamlit
        st.pyplot(plt)
        plt.clf()  # Limpa a figura após exibição