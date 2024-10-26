# Como rodar:
# pip install streamlit, streamlit-cropper scipy, numpy, matplotlib, opencv-python
# streamlit run interface.py
#NT:(2322434)mod4 = 2  ->  Momentos invariantes de Hu, se NT=2
#Eric Miranda: 771803
#Gustavo Lorenzo: 782633
#Vinicius Augusto: 767998
import numpy as np
from skimage.feature import graycomatrix, graycoprops
#from scipy.stats import entropy opcional, escolher qual processa mais rapido a entropia -> fiz na mão
import cv2  # Adicione cv2 para calcular os momentos de Hu

def circular_offsets(distance):
    angles = []
    for x in range(-distance, distance + 1):
        for y in range(-distance, distance + 1):
            if np.round(np.sqrt(x**2 + y**2)) == distance and (x, y) != (0, 0):
                angles.append(np.arctan2(y, x))  # Mudado para formato radiano
    return angles

def glcm(image, distances, gray_levels=256, symmetric=True, normed=True):
    for distance in distances:
            image = np.copy(image)

            # obter deslocamentos circulares para a distância atual(reza para estar certo -> nos meus calculos deu certo *u* )
            
            #Deslocamento Circular
            angles = circular_offsets(distance)
            #GLCM
            glcm = graycomatrix(image, [distance], angles, gray_levels, symmetric, normed)

            #Homogeneidade
            homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
            print(f"Homogeneidade para distancia: {distance} and angle:{angles}=: {homogeneity}")

            # Entropia
            glcm_probabilities = glcm / glcm.sum()
            glcm_probabilities_flat = glcm_probabilities.flatten()
            glcm_probabilities_flat = glcm_probabilities_flat[glcm_probabilities_flat > 0]
            # Calculate Entropy using the formula: -∑ S(L, M) log2(S(L, M))
            glcm_entropy = -np.sum(glcm_probabilities_flat * np.log2(glcm_probabilities_flat))# a funcao da biblioteca scipy.stats faria o mesmo calculo
            #escolher qual custa menos tempo p processar
            #caso mude, use essa linha de codigo: glcm_entropy = entropy(glcm_probabilities_flat, base=2)   #estou definindo a base 2 por ser entropia em bit
            print(f"Entropia para distancia: {distance} and angle:{angles}= {glcm_entropy}")

            #a cv2.moment aceita somente uma matriz de dimensao 2d, a glcm e 4d
            glcm_2d = np.sum(glcm, axis=(2, 3))
            # Momentos Invariantes de Hu
            # Normalizar GLCM para ser uma imagem de intensidade
            glcm_normalized = (glcm_2d  / glcm_2d.max() * 255).astype(np.uint8)

            #Calcular momento invariante de Hu
            moments = cv2.moments(glcm_normalized)
            hu_moments = cv2.HuMoments(moments).flatten()

            print(f"Momentos Invariantes de Hu para distancia: {distance} and angle: {angles}:")
            for i, hu_moment in enumerate(hu_moments, 1):
                print(f"Hu[{i}]: {hu_moment}")
    return glcm































