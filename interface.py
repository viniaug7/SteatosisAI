import streamlit as st
from PIL import Image

# Título da aplicação
st.title("Visualizador de Imagens")

# Permite que o usuário faça o upload de um arquivo
uploaded_file = st.file_uploader("Escolha uma imagem", type=["png", "jpg", "jpeg"])

# Verifica se o arquivo foi enviado
if uploaded_file is not None:
    # Abre a imagem utilizando o Pillow (PIL)
    image = Image.open(uploaded_file)

    # Exibe a imagem na tela
    st.image(image, caption="Imagem enviada.", use_column_width=True)
