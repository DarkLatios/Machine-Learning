import cv2
import numpy as np
from matplotlib import pyplot as plt
vc=cv2.VideoCapture(0)
while(vc.isOpened()):
    ret,img=vc.read()
    cv2.imshow('Output',img)
    img2 =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    x=150
    imgthreshold=cv2.inRange(img,cv2.cv.Scalar(x,x,x),cv2.cv.Scalar(255,255,255),)
    cv2.imshow('threshold',imgthreshold)
    
    
    edges=cv2.Canny(imgthreshold,100,200)
    cv2.imshow('Filter',edges)

    roi = img [80:120, 0:320]
    roi2 = img [100:140, 0:320]
    roi3 = img [120:160, 0:320]
    img1 = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)
    img3 = cv2.cvtColor(roi3, cv2.COLOR_BGR2GRAY)
    ret, img1 = cv2.threshold(img1, 0, 255, 0)
    cv2.bitwise_not(img1, img1)
    ret, img2 = cv2.threshold(img2, 0, 255, 0)
    cv2.bitwise_not(img2, img2)
    ret, img3 = cv2.threshold(img3, 0, 255, 0)
    cv2.bitwise_not(img3, img3)

    Eerode = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    Ddilate = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    cv2.erode(img1, Eerode)
    cv2.dilate(img1, Ddilate)
    cv2.erode(img2, Eerode)
    cv2.dilate(img2, Ddilate)
    cv2.erode(img3, Eerode)
    cv2.dilate(img3, Ddilate)

    contours, hierarchy = cv2.findContours(img1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours2, hierarchy = cv2.findContours(img2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours3, hierarchy = cv2.findContours(img3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for i in contours :
        print(i);
        area = cv2.contourArea(i)
        if area>1000 :
            
            epsilon = cv2.arcLength(i,True)
            approx  = cv2.approxPolyDP(i, 0.02*epsilon, True)
            if len(approx)>= 4:
                scnt=approx
                x,y,w,h=cv2.boundingRect(scnt)
                cv2.rectangle(img, (x, y+80), (x+w, y+h), (0,255,0), 2)
                break

    for i in contours2 :
        area = cv2.contourArea(i)
        if area>1000 :

            epsilon = cv2.arcLength(i,True)
            approx  = cv2.approxPolyDP(i, 0.02*epsilon, True)
            if len(approx)>= 4:
                scnt=approx
                x,y,w,h=cv2.boundingRect(scnt)
                cv2.rectangle(img, (x, y+100), (x+w, y+h), (0,255,0), 2)
                break
    cv2.imshow("Frame", img)
    
            

            
            
            
            
        
    
    



    k=cv2.waitKey(10)
    if k==27:
        break

vc.release()
cv2.destroyAllWindows()
    
    



