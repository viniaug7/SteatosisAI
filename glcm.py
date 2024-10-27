# Como rodar:
# pip install streamlit, streamlit-cropper scipy, numpy, matplotlib
# streamlit run interface.py
#NT:(2322434)mod4 = 2  ->  Momentos invariantes de Hu, se NT=2
#Eric Miranda: 771803
#Gustavo Lorenzo: 782633
#Vinicius Augusto: 767998
import numpy as np
import streamlit as st
from skimage.feature import graycomatrix, graycoprops
#from scipy.stats import entropy opcional, escolher qual processa mais rapido a entropia -> fiz na mão
import cv2  # Adicione cv2 para calcular os momentos de Hu

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
            print(f"Homogeneidade para distancia: {distance}=: {homogeneity}")

            # Entropia
            glcm_probabilities = glcm / glcm.sum()
            glcm_probabilities_flat = glcm_probabilities.flatten()
            glcm_probabilities_flat = glcm_probabilities_flat[glcm_probabilities_flat > 0]
            # Calculate Entropy using the formula: -∑ S(L, M) log2(S(L, M))
            glcm_entropy = -np.sum(glcm_probabilities_flat * np.log2(glcm_probabilities_flat))# a funcao da biblioteca scipy.stats faria o mesmo calculo
            #escolher qual custa menos tempo p processar
            #caso mude, use essa linha de codigo: glcm_entropy = entropy(glcm_probabilities_flat, base=2)   #estou definindo a base 2 por ser entropia em bit
            print(f"Entropia para distancia: {distance}= {glcm_entropy}")

            #a cv2.moment aceita somente uma matriz de dimensao 2d, a glcm e 4d
            glcm_2d = np.sum(glcm, axis=(2, 3))
            # Momentos Invariantes de Hu
            # Normalizar GLCM para ser uma imagem de intensidade
            glcm_normalized = (glcm_2d  / glcm_2d.max() * 255).astype(np.uint8)

            #Calcular momento invariante de Hu
            moments = cv2.moments(glcm_normalized)
            hu_moments = cv2.HuMoments(moments).flatten()

            print(f"Momentos Invariantes de Hu para distancia: {distance}:")
            for i, hu_moment in enumerate(hu_moments, 1):
                print(f"Hu[{i}]: {hu_moment}")
    return glcm































