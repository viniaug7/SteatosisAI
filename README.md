# Sistema Auxiliar de Diagnóstico de Esteatose Hepática  

Este projeto apresenta um **sistema auxiliar para diagnóstico de Esteatose Hepática Não Alcoólica (NAFLD)** utilizando **imagens ultrassonográficas** e classificadores de aprendizado de máquina. Foram implementados dois modelos principais:  

- **Support Vector Machine (SVM)** com descritores de textura (GLCM e Momentos Invariantes de Hu).  
- **VGG16 (CNN profunda)** com *fine-tuning* para classificação binária (saudável / esteatose).  

Além da implementação dos modelos, o sistema conta com uma **interface gráfica interativa em Streamlit**, permitindo que usuários sem conhecimento avançado em Python utilizem a ferramenta diretamente pelo navegador.  

---

## Funcionalidades  

- **Pré-processamento de imagens**: recorte de ROIs (fígado e rim), cálculo do índice hepatorenal (HI) e normalização.  
- **Extração de características**: GLCM (descritores de Haralick) e Momentos Invariantes de Hu.  
- **Classificação**:  
  - 🔹 **SVM** (rápido, baseado em descritores manuais).  
  - 🔹 **VGG16** (aprendizado profundo com transfer learning).  
- **Interface web (Streamlit)**:  
  - Visualização de imagens e ROIs.  
  - Seleção e corte interativo de regiões.  
  - Treinamento e classificação diretamente via navegador.  
- **Comparação de desempenho** entre os classificadores. 
