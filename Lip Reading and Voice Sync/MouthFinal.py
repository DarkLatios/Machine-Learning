import cv2
import sys
import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
mouth_cascade = cv2.CascadeClassifier('C:\Python 2.7\haarcascade_mcs_mouth.xml')
video_capture = cv2.VideoCapture(0)




def read_images(path, sz=(256,256)):
    
    
    X,y = [], []
    folder_names = []
    default='Unknown'
    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            default='Unknown'
            folder_names.append(subdirname)
            subject_path = os.path.join(dirname, subdirname)
            for filenames in os.listdir(subject_path):
                try:
                    im = cv2.imread(os.path.join(subject_path, filenames), cv2.IMREAD_GRAYSCALE)
                    nbr = int(os.path.split(filenames)[1].split(".")[0].replace("Image", ""))
                    
                    if (sz is not None):
                        im = cv2.resize(im, sz)
                    X.append(np.asarray(im, dtype=np.uint8))
                    y.append(nbr)
                except IOError, (errno, strerror):
                    print "I/O error({0}): {1}".format(errno, strerror)
                except:
                    print "Unexpected error:", sys.exc_info()[0]
                    raise
                nbr=nbr+1
        
    return [X,y,folder_names]

    
    

pathdir='C:\Users\user\Desktop\Trained Images/'
quanti = int(raw_input('Number:'))
for i in range(quanti):
    print('HELLO USER '+str(i+1)+' what is your name?')
    nome = raw_input('Name:')
    if not os.path.exists(pathdir+nome): os.makedirs(pathdir+nome)
    print(str(nome)+'Collecting Specimens,Please Look At me!!!! ')
    while (1):
        ret,frame = video_capture.read(0)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 3)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.imshow('Recognition',frame)

        
        if cv2.waitKey(10) == ord('s'):
            break
    cv2.destroyAllWindows()




    
    start = time.time()
    count = 0
    count2=0
    while int(time.time()-start) <=12:
        
        ret,img = video_capture.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            try:
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = img[y:y+h, x:x+w]
                resize=cv2.resize(roi_color,(256,256))
                #cv2.imshow('face',resize)
                mouth = mouth_cascade.detectMultiScale(resize,1.4,22,minSize=(25,25))
                for(mx,my,mw,mh) in mouth:
                    count +=1
                    cv2.rectangle(resize, (mx, my), ((mx+mw), my+mh), (0, 255, 0), 2)
                    cv2.imshow('mouth',resize)
                    resize_m=cv2.resize(resize[my:my+mh,mx:mx+mw],(256,256))
                    cv2.imshow('Lips',resize_m)
                    #cv2.rectangle(roi_color, (mx-b, my-a), (mx+mw+b, my-a+mh+a), (0, 255, 0), 2)
                    #cv2.putText(frame,'Click!', (x,y), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,250),3,1)
                    #resize=cv2.resize(img[y:y+h, x:x+w],(256,256))
                    #resized_image = cv2.resize(frame[y:y+h,x:x+w], (256, 256))
                    if count%5 == 0:
                        
                        print  (pathdir+nome+str(count2)+'.jpg')
                        cv2.imwrite( pathdir+nome+'/'+str(count2)+'.jpg', resize_m );
                        count2+=1
                        cv2.imshow('Recognition',img)
                        
            except KeyError:
                pass
            
        cv2.waitKey(10)
    cv2.destroyAllWindows()

                      

video_capture.release()
cv2.destroyAllWindows()
