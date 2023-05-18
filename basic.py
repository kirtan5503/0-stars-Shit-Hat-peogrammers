import cv2
import numpy as np
import face_recognition


imgElon= face_recognition.load_image_file('imagesbasic/Elonmusk2.jpg')
imgElon = cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)
imgElontest= face_recognition.load_image_file('imagesbasic/Billgates.jpg')
imgElontest = cv2.cvtColor(imgElontest,cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgElon)[0]
encodElon = face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon,(faceLoc[3],faceLoc[0],faceLoc[1],faceLoc[2]),(333),2)

faceLoc = face_recognition.face_locations(imgElontest)[0]
encodElontest = face_recognition.face_encodings(imgElontest)[0]
cv2.rectangle(imgElontest,(faceLoc[3],faceLoc[0],faceLoc[1],faceLoc[2]),(333),2)

results = face_recognition.compare_faces([encodElon],encodElontest)
print(results)


cv2.imshow('Elon Musk',imgElon)
cv2.imshow('Elon Musk Test',imgElontest)
cv2.waitKey(0)