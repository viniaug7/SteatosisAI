# Como rodar:
# pip install streamlit, scipy, numpy
# streamlit run interface.py
from io import BytesIO
import streamlit as st
import scipy.io
import scipy.misc
import numpy as np
import io
from st_clickable_images import clickable_images
from PIL import Image
import base64




# d
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



def transformar_imagens_mat_em_itens_sidebar(arquivo_mat):
    # Carregar arquivo .mat
    data = scipy.io.loadmat(arquivo_mat)
    data_array = data["data"]
    images = data_array["images"]
    # pegar uma imagem == imagem = images[0][n][m]
    # retornar todas as imagens (flatmap a coisa acima)
    #return [images[0][n][m] for n in range(55) for m in range(10)]
    # criar itens na sidebar
    for n in range(55):
        imagens = []
        for m in range(10):
            imagem = images[0][n][m]
            img = Image.fromarray(imagem)
            byte_io = BytesIO()
            img.save(byte_io, format='JPEG')
            image_bytes = byte_io.getvalue()
            encoded = base64.b64encode(image_bytes).decode()
            imagens.append(f"data:image/jpeg;base64,{encoded}")

        with st.sidebar.expander(f"Paciente {n+1}"):
            # print(imagens[0])
            # st.image(imagens[0], caption="Imagem 0", width=150)
            clicked = clickable_images(
                imagens,
                titles=[f"Image #{str(i)}" for i in range(len(images))],
                # div_style={"display": "flex", "justify-content": "center", "flex-wrap": "wrap"},
                # img_style={"margin": "5px", "height": "200px"},
            )
            print(clicked)


# #E assim que se inicia as pastas
#pastas = {
#    "Pasta 1": ["https://via.placeholder.com/150", "https://via.placeholder.com/200"],
#    "Pasta 2": ["https://via.placeholder.com/250", "https://via.placeholder.com/300"],
#    "Pasta 3": ["https://via.placeholder.com/350", "https://via.placeholder.com/400"],
#}
##E assim que se adiciona pastas
#pastas["Pasta4"] = ["https://via.placeholder.com/450", "https://via.placeholder.com/500"]
#pastas["Pasta5"] = ["https://via.placeholder.com/550", "https://via.placeholder.com/600"]
##E assim que se adiciona uma foto em uma pasta
#pastas["Pasta 1"].append("https://via.placeholder.com/650")
#Nome para aplicacao
st.title("Trabalho PAI")

# Upload de arquivo no topo do sidebar
st.sidebar.title("Adicionar Foto")
arquivo = st.sidebar.file_uploader("Carregar uma imagem", type=["mat", "jpeg", "jpg", "png"])

#aqui é onde vamos receber as imagens (esta no sidebar)
if arquivo:
    imagem= None
    # Verifica o tipo de arquivo
    if arquivo.type in ["image/jpeg", "image/png"]:
        imagem = Image.open(arquivo)
        imagem_url = st.sidebar.image(imagem, caption="Imagem Carregada", width=150)
    elif arquivo.type == "application/octet-stream":  # Para arquivos .mat
        transformar_imagens_mat_em_itens_sidebar(arquivo)
        # imagem_url = st.sidebar.image(imagem, caption="Imagem Carregada (MAT)", width=150)

    # if imagem:
    #     nome_pasta = f"Pasta {len(pastas) + 1}"
    #     if nome_pasta not in pastas:
    #         pastas[nome_pasta] = []
    #     if imagem not in pastas[nome_pasta]:
    #         pastas[nome_pasta].append(imagem)
            

imagem_selecionada = None
#para cada pasta no dicionario(paciente), cria um expande(printar no sidebar)
# for nome_pasta, fotos in pastas.items():
#     with st.sidebar.expander(nome_pasta):
#         st.write(f"Fotos da {nome_pasta}:")
#         # Para cada foto dentro da pasta, exibe a imagem
#         for i, foto in enumerate(fotos):
#             if isinstance(foto, Image.Image):  # Se for objeto Image
#                 st.image(foto, width=100)
#                 if st.button(f"Selecionar Foto {i+1} de {nome_pasta}", key=f"botao_{nome_pasta}_{i}"):
#                     imagem_selecionada = foto
#             else:  # Se for URL
#                 st.image(foto, width=100)
#                 if st.button(f"Selecionar {foto}", key=foto):
#                     imagem_selecionada = foto

if imagem_selecionada:
    st.write("Imagem selecionada:")
    st.image(imagem_selecionada, width=300)
else:
    st.write("Nenhuma imagem selecionada.")
st.write("Aqui é a página principal. As fotos estão nas pastas da sidebar.")
