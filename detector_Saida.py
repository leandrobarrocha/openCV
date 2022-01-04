import cv2
import pandas as pd
import numpy as np
from datetime import datetime


detectorFace = cv2.CascadeClassifier('haarcascade-frontalface-default.xml')
reconhecedor = cv2.face.LBPHFaceRecognizer_create(2, 2, 15, 15, 70)
reconhecedor.read('trainingData.yml')
data = pd.read_csv("registro.csv")
largura, altura = 170, 170
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
data_e_hora_atuais = datetime.now()
data_e_hora_em_texto = data_e_hora_atuais.strftime('%d/%m/%Y %H:%M')
camera =cv2.VideoCapture(1)

#Converte a inmágem de entrada em escalka de cinza para aplicar os filtros de seleção
while(True):
    conectado, imagem = camera.read()
    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    facesDetectadas = detectorFace.detectMultiScale(imagemCinza, scaleFactor=1.5, minSize=(150, 150))

    #identificação de faces com banco interno (possível fazer uma conecção com banco de dados) 
    for (x, y, l, a) in facesDetectadas:
        imagemFace = cv2.resize(imagemCinza[y:y + a, x:x + l], (largura, altura))
       #padrão para desenhar um retangulo envolta da face onde será feito o reconhecimento de faces
        cv2.rectangle(imagem, (x, y), (x + l, y + a), (255, 0, 255), 2)
        id, confianca = reconhecedor.predict(imagemFace)
        print(confianca)
        print(id)
        nome = ""
        if id == 1:
            nome = "Leandro"
            cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 255, 0), 2)
            cv2.putText(imagem, nome, (x, y + (a + 30)), font, 2, (0, 255, 0))
            cv2.putText(imagem, str(confianca), (x, y + (a + 50)), font, 1, (0, 255, 0))
        else:
            nome = "Desconhecido"
            cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 0, 255), 2)
            cv2.putText(imagem, nome, (x, y + (a + 30)), font, 2, (0, 0, 255))
            cv2.putText(imagem, str(confianca), (x, y + (a + 50)), font, 1, (0, 0, 255))


        # selecionando o rosto detectado que está sem data de saida
        registro3 = data.loc[data['id'] == id & (data['data_saida'].isnull())]
        print(registro3)

        # identificar o index
        inde = registro3.index.values.tolist()
        # print(inde)

        # so colocar os valores faltantes
        registro3['data_saida'] = registro3['data_saida'].map({np.nan: data_e_hora_em_texto}, na_action=None)
        registro3['confianca_saida'] = registro3['confianca_saida'].map({np.nan: confianca}, na_action=None)
        print(registro3)

        # excluindo a linha com valor faltante
        data.drop(inde, inplace=True)
        #print(data2)

        # juntando os dois bancos de dados
        data = data.append(registro3)
        print(data)

        # salvando o banco de dados
        data.to_csv('registro.csv', sep=',', encoding='utf-8')

        cv2.destroyAllWindow()

    cv2.imshow('Face', imagem)






    if cv2.waitKey(1) == ord('s'):
        break

camera.release()
cv2.destroyAllWindow()
