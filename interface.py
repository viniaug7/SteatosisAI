import streamlit as st
from PIL import Image
from scipy.io import loadmat
import io

# Função para salvar e retornar a imagem
def carregar_imagem(arquivo):
    file_extension = arquivo.name.split(".")[-1].lower()

    if file_extension == 'mat':
        mat_data = loadmat(arquivo)
        return f"Arquivo .mat carregado: {arquivo.name}", None
    else:
        image = Image.open(arquivo)
        return arquivo.name, image

# Interface principal
st.title("Upload e Visualização de Imagens")
st.write("Carregue imagens no formato .mat, .png, .jpeg ou .jpg para visualizá-las quando selecionadas.")

# Armazenamento de arquivos
if 'pastas' not in st.session_state:
    st.session_state['pastas'] = {"a", "b", "c"}

# Upload de arquivo
uploaded_file = st.file_uploader("Carregar Imagem", type=["mat", "png", "jpeg", "jpg"])

# Se houver um arquivo enviado, ele será processado
if uploaded_file:
    nome_arquivo, imagem = carregar_imagem(uploaded_file)
    if imagem is not None:
        # Pedir ao usuário para escolher ou criar uma nova pasta
        pasta = "Foto_individual"
        
        if pasta:
            # Se a pasta não existir, cria uma nova
            if pasta not in st.session_state['pastas']:
                print(st.session_state['pastas'])
                print(st.session_state['pastas'][pasta])
                st.session_state['pastas'][pasta] = []
            # Adiciona a imagem à pasta correspondente
            st.session_state['pastas'][pasta].append((nome_arquivo, imagem))
    else:
        #VAMOS TER QUE TRATAR OS PACIENTES POR PASTAS
        st.sidebar.write(nome_arquivo)  # Para arquivos .mat, só mostra a mensagem
uploaded_file = None
# Exibição de pastas e imagens no sidebar
st.sidebar.write("Pastas carregadas:")
with st.sidebar:
    for a in st.session_state['pastas']:
        with st.expander(a, True):
            #definir um vetor de imagens e colocar aqui

            """st.title("🎈 Okld's Gallery")

        with st.expander("✨ APPS", True):
            page.item("Streamlit gallery", apps.gallery, default=True)

        with st.expander("🧩 COMPONENTS", True):
            page.item("Ace editor", components.ace_editor)
            page.item("Disqus", components.disqus)
            page.item("Elements⭐", components.elements)
            page.item("Pandas profiling", components.pandas_profiling)
            page.item("Quill editor", components.quill_editor)
            page.item("React player", components.react_player)"""

#AQUI ESTA O PROBLEMA
#-----------------------------------------------------------
"""
pasta_selecionada = st.sidebar.selectbox("Selecione uma pasta", list(st.session_state['pastas'].keys()))

# Se uma pasta foi selecionada, mostre as imagens dela
if pasta_selecionada:
    imagens_pasta = st.session_state['pastas'][pasta_selecionada]
    nomes_imagens = [img[0] for img in imagens_pasta]
    
    # Selecionar uma imagem da pasta
    imagem_selecionada = st.sidebar.selectbox("Selecione uma imagem para visualizar", nomes_imagens)
    
    # Mostrar a imagem selecionada
    for nome, img in imagens_pasta:
        if nome == imagem_selecionada:
            st.image(img, caption=f"Imagem: {nome}", use_column_width=True)
            break
#-----------------------------------------------------------"""