import streamlit as st
import scipy.io
import numpy as np
import io
from PIL import Image
def carregar_imagem_mat(arquivo_mat):
    # Carregar arquivo .mat
    mat_data = scipy.io.loadmat(arquivo_mat)
    # Supõe-se que a imagem esteja armazenada sob uma chave específica (ajuste conforme necessário)
    chave_imagem = list(mat_data.keys())[-1]
    imagem = mat_data[chave_imagem]
    
    # Normalizar se necessário
    if imagem.max() > 1:
        imagem = imagem / 255.0

    return imagem
#E assim que se inicia as pastas
pastas = {
    "Pasta 1": ["https://via.placeholder.com/150", "https://via.placeholder.com/200"],
    "Pasta 2": ["https://via.placeholder.com/250", "https://via.placeholder.com/300"],
    "Pasta 3": ["https://via.placeholder.com/350", "https://via.placeholder.com/400"],
}
#E assim que se adiciona pastas
pastas["Pasta4"] = ["https://via.placeholder.com/450", "https://via.placeholder.com/500"]
pastas["Pasta5"] = ["https://via.placeholder.com/550", "https://via.placeholder.com/600"]
#E assim que se adiciona uma foto em uma pasta
pastas["Pasta 1"].append("https://via.placeholder.com/650")
#Nome para aplicacao
st.title("Trabalho PAI")

# Upload de arquivo no topo do sidebar
st.sidebar.title("Adicionar Foto")
arquivo = st.sidebar.file_uploader("Carregar uma imagem", type=["mat", "jpeg", "jpg", "png"])

#aqui é onde vamos receber as imagens (esta no sidebar)
if arquivo:
    # Verifica o tipo de arquivo
    if arquivo.type == "image/jpeg" or arquivo.type == "image/png":
        imagem = Image.open(arquivo)
        st.sidebar.image(imagem, caption="Imagem Carregada", width=150)
    elif arquivo.type == "application/octet-stream":  # Para arquivos .mat
        imagem = carregar_imagem_mat(arquivo)
        st.sidebar.image(imagem, caption="Imagem Carregada (MAT)", width=150)

imagem_selecionada = None
#para cada pasta no dicionario(paciente), cria um expande(printar no sidebar)
for nome_pasta, fotos in pastas.items():
    with st.sidebar.expander(nome_pasta):
        st.write(f"Fotos da {nome_pasta}:")
        # Para cada foto dentro da pasta, exibe a imagem
        for foto in fotos:
            if st.button(f"Selecionar {foto}", key=foto):
                imagem_selecionada = foto

if imagem_selecionada:
    st.write("Imagem selecionada:")
    st.image(imagem_selecionada, width=300)
else:
    st.write("Nenhuma imagem selecionada.")
st.write("Aqui é a página principal. As fotos estão nas pastas da sidebar.")
