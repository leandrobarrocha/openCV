import cv2
import csv
from datetime import datetime


detectorFace = cv2.CascadeClassifier('haarcascade-frontalface-default.xml')
reconhecedor = cv2.face.LBPHFaceRecognizer_create(2, 2, 15, 15, 70)
reconhecedor.read('trainingData.yml')
largura, altura = 220, 220
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
data_e_hora_atuais = datetime.now()
data_e_hora_em_texto = data_e_hora_atuais.strftime('%d/%m/%Y %H:%M')
camera =cv2.VideoCapture(1)

#Converte a inmágem de entrada em escalka de cinza para aplicar os filtros de seleção
while(True):
    conectado, imagem = camera.read()
    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    facesDetectadas = detectorFace.detectMultiScale(imagemCinza, scaleFactor=1.6, minSize=(150, 150))

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

        if confianca > 70 and id == 1:
            with open('registro.csv', 'at') as arquivo_csv:
                colunas = ['id', 'data_entrada', 'confianca_entrada', 'data_saida', 'confianca_saida', 'pessoa']
                escrever = csv.DictWriter(arquivo_csv, fieldnames=colunas, delimiter=',', lineterminator='\n')
                escrever.writeheader()
                escrever.writerow({'id': id, 'data_entrada': data_e_hora_em_texto, 'confianca_entrada': confianca, 'pessoa': nome})
                print('Gravado com sucesso')
                cv2.destroyAllWindow()




    cv2.imshow('Face', imagem)






    if cv2.waitKey(1) == ord('s'):
        break

camera.release()
cv2.destroyAllWindow()
print('Registro salvo')
