# Como rodar:
# pip install streamlit, streamlit-cropper scipy, numpy, matplotlib
# streamlit run interface.py
#NT:(2322434)mod4 = 2  ->  Momentos invariantes de Hu, se NT=2
#Gustavo Lorenzo Campos: 782633
#Eric Miranda: 771803
#Vinicius Augusto: 767998
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import entropy

def glcm(image, distances, angles, gray_levels=256, symmetric=True, normed=True):

    image = np.copy(image)

    glcm = graycomatrix(image, distances, angles, gray_levels, symmetric, normed)

    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    print(f"Homogeneidade: {homogeneity}")

     # Normalize GLCM to get probabilitiesDixgracaaaaaaaaaaaaaaaaaa refiz esse codigo 5x
    glcm_probabilities = glcm / glcm.sum()
    
    # Flatten GLCM probabilities to calculate entropy
    glcm_probabilities_flat = glcm_probabilities.flatten()
    
    # Filter out zero probabilities to avoid log(0) errors
    glcm_probabilities_flat = glcm_probabilities_flat[glcm_probabilities_flat > 0]
    
    # Calculate Entropy using the formula: -∑ S(L, M) log2(S(L, M))
    glcm_entropy = -np.sum(glcm_probabilities_flat * np.log2(glcm_probabilities_flat))
    print(f"entropia:{glcm_entropy}")

    return glcm































