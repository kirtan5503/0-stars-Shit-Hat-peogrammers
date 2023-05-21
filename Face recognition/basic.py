import cv2
import numpy as np
import face_recognition


img1= face_recognition.load_image_file('imagesbasic/Billgates.jpg')
img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
img2= face_recognition.load_image_file('imagesbasic/kirtan photo1.jpg')
img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(img1)[0]
encod1 = face_recognition.face_encodings(img1)[0]
cv2.rectangle(img1,(faceLoc[3],faceLoc[0],faceLoc[1],faceLoc[2]),(333),2)

faceLoc = face_recognition.face_locations(img2)[0]
encod2 = face_recognition.face_encodings(img2)[0]
cv2.rectangle(img2,(faceLoc[3],faceLoc[0],faceLoc[1],faceLoc[2]),(333),2)

results = face_recognition.compare_faces([encod1],encod2)
facedis = face_recognition.face_distance([encod1],encod2)
print(results,facedis)
cv2.putText(img2,f'{results}{round(facedis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),2)


cv2.imshow('Elon Musk',img1)
cv2.imshow('Elon Musk Test',img2)
cv2.waitKey(0)