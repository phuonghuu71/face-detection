import cv2
import numpy as np
from PIL import Image
import sqlite3
import argparse
import math
import time

import os

import tkinter as tk
from tkinter import messagebox

from pathlib import Path
import glob


def register():
    cam = cv2.VideoCapture(0)
    detector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    def insertOrUpdate(Id,Name):
        conn=sqlite3.connect('FaceBase.db')
        cmd="SELECT * FROM People WHERE ID= '"+str(Id)+"'"
        cursor=conn.execute(cmd)
        isRecordExist=0
        for row in cursor:
            isRecordExist=1
        if(isRecordExist==1):
            cmd="UPDATE People SET Name='"+str(Name)+"' WHERE ID='"+str(Id)+"'"
        else:
            cmd="INSERT INTO People(Id,Name) Values('"+str(Id)+"','"+str(Name)+"')"
        conn.execute(cmd)
        conn.commit()
        conn.close()


    id=id_input.get()
    name=name_input.get()
    insertOrUpdate(id,name)
    sampleNum=0
    while(True):
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            
            #incrementing sample number 
            sampleNum=sampleNum+1
            #saving the captured face in the dataset folder
            cv2.imwrite("dataSet/User."+id +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])

            cv2.imshow('frame',img)
        #wait for 100 miliseconds 
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
        # break if the sample number is morethan 20
        elif sampleNum>60:
            break
    cam.release()
    cv2.destroyAllWindows()

    training()

    #get data from sqlite by ID
    def getProfile(id):
        conn=sqlite3.connect("FaceBase.db")
        cmd="SELECT * FROM People WHERE ID="+str(id)
        cursor=conn.execute(cmd)
        profile=None
        for row in cursor:
            profile=row
        conn.close()
        return profile

    # raiseFrame(RegisterFrame)

def training():
    
    recognizer = cv2.face_LBPHFaceRecognizer.create()
    # recognizer = cv2.fa
    path='dataSet'

    def getImagesAndLabels(path):
        #get the path of all the files in the folder
        imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
        faces=[]
        IDs=[]
        for imagePath in imagePaths:
            faceImg=Image.open(imagePath).convert('L')
            faceNp=np.array(faceImg,'uint8')
            #split to get ID of the image
            ID=int(os.path.split(imagePath)[-1].split('.')[1])
            faces.append(faceNp)
            print(ID)
            IDs.append(ID)
            cv2.imshow("traning",faceNp)
            cv2.waitKey(10)
        return IDs, faces

    Ids,faces=getImagesAndLabels(path)
    #trainning
    recognizer.train(faces,np.array(Ids))
    recognizer.save('recognizer\\trainningData.yml')
    cv2.destroyAllWindows()

    # raiseFrame(TrainingFrame)

def recog_face_gender_age():
    faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    rec=cv2.face.LBPHFaceRecognizer_create()

    rec.read("recognizer\\trainningData.yml")
    id=0
    #set text style
    fontface = cv2.FONT_HERSHEY_SIMPLEX
    fontscale = 1
    fontcolor = (203,23,252)

    #get data from sqlite by ID
    def getProfile(id):
        conn=sqlite3.connect("FaceBase.db")
        cmd="SELECT * FROM People WHERE ID="+str(id)
        cursor=conn.execute(cmd)
        profile=None
        for row in cursor:
            profile=row
        conn.close()
        return profile

    def getFaceBox(net, frame, conf_threshold=0.7):
        frameOpencvDnn = frame.copy()
        frameHeight = frameOpencvDnn.shape[0]
        frameWidth = frameOpencvDnn.shape[1]
        blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

        net.setInput(blob)
        detections = net.forward()
        bboxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                x1 = int(detections[0, 0, i, 3] * frameWidth)
                y1 = int(detections[0, 0, i, 4] * frameHeight)
                x2 = int(detections[0, 0, i, 5] * frameWidth)
                y2 = int(detections[0, 0, i, 6] * frameHeight)
                bboxes.append([x1, y1, x2, y2])
                cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
        return frameOpencvDnn, bboxes


    parser = argparse.ArgumentParser(description='Use this script to run age and gender recognition using OpenCV.')
    parser.add_argument('--input', help='Path to input image or video file. Skip this argument to capture frames from a camera.')

    args = parser.parse_args()

    faceProto = "opencv_face_detector.pbtxt"
    faceModel = "opencv_face_detector_uint8.pb"

    ageProto = "age_deploy.prototxt"
    ageModel = "age_net.caffemodel"

    genderProto = "gender_deploy.prototxt"
    genderModel = "gender_net.caffemodel"

    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    genderList = ['Male', 'Female']

    # Load network
    ageNet = cv2.dnn.readNet(ageModel, ageProto)
    genderNet = cv2.dnn.readNet(genderModel, genderProto)
    faceNet = cv2.dnn.readNet(faceModel, faceProto)

    # Open a video file or an image file or a camera stream
    cap = cv2.VideoCapture(args.input if args.input else 0)
    padding = 20
    while cv2.waitKey(1) < 0:
        # Read frame
        t = time.time()
        hasFrame, frame = cap.read()
        
        #ret,img=cam.read()
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces=faceDetect.detectMultiScale(gray,1.3,5)

        if not hasFrame:
            cv2.waitKey()
            break

        frameFace, bboxes = getFaceBox(faceNet, frame)
        #frameFace, bboxes = faces
        if not bboxes:
            print("No face Detected, Checking next frame")
            continue
        
        for(x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            id,conf=rec.predict(gray[y:y+h,x:x+w])
            profile=getProfile(id)
            if(profile!=None):
                cv2.putText(frameFace, "Name: " + str(profile[1]), (x, y-40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

        for bbox in bboxes:
            
            face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]

            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            gender = genderList[genderPreds[0].argmax()]
            
            print("Gender : {}, conf = {:.3f}".format(gender, genderPreds[0].max()))

            ageNet.setInput(blob)
            agePreds = ageNet.forward()
            age = ageList[agePreds[0].argmax()]
            print("Age Output : {}".format(agePreds))
            print("Age : {}, conf = {:.3f}".format(age, agePreds[0].max()))

            label = "{},{}".format(gender, age)
            cv2.putText(frameFace, label, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Age Gender Demo", frameFace)
        print("time : {:.3f}".format(time.time() - t))


#Tkinter
root = tk.Tk()
root.title("Recognition face, gender, age")
#Frames
RegisterFrame=tk.Frame(root)
TrainingFrame=tk.Frame(root)
recog_face_gender_ageFrame = tk.Frame(root)

#Define Frame List
frameList=[RegisterFrame, TrainingFrame, recog_face_gender_ageFrame]
#Configure all Frames
for frame in frameList:
	frame.grid(row=0,column=0, sticky='news')
	frame.configure(bg='white')
	
def raiseFrame(frame):
	frame.tkraise()

def RegisterFrameRaiseFrame():
	raiseFrame(RegisterFrame)
#def TrainingFrameRaiseFrame():
#	raiseFrame(TrainingFrame)
def recog_face_gender_ageFrameRaiseFrame():
    raiseFrame(recog_face_gender_ageFrame)
#Tkinter Vars
#Stores user's name when registering
id_input = tk.StringVar()
name_input = tk.StringVar()

tk.Label(recog_face_gender_ageFrame,text="Recognition",font=("Courier", 60),bg="white").grid(row=1,column=2,columnspan=5)
RegisterButton = tk.Button(recog_face_gender_ageFrame,text="Register",bg="white",font=("Arial", 30),command=RegisterFrameRaiseFrame)
RegisterButton.grid(row=3,column=1)

#TrainButton = tk.Button(recog_face_gender_ageFrame,text="Training",command=training,bg="white",font=("Arial", 30))
#TrainButton.grid(row=3,column=4)

RecognitionButton = tk.Button(recog_face_gender_ageFrame,text="Recognition",command=recog_face_gender_age,bg="white",font=("Arial", 30))
RecognitionButton.grid(row=3,column=7)

tk.Label(RegisterFrame,text="Register",font=("Courier",60),bg="white").grid(row=1,column=2)
tk.Label(RegisterFrame,text="ID",font=("Courier",60),bg="white").grid(row=2,column=1)
tk.Label(RegisterFrame,text="Name: ",font=("Arial",30),bg="white").grid(row=3,column=1)
idEntry = tk.Entry(RegisterFrame, textvariable=id_input,font=("Arial",30)).grid(row=2,column=2)
nameEntry=tk.Entry(RegisterFrame,textvariable=name_input,font=("Arial",30)).grid(row=3,column=2)

regButton = tk.Button(RegisterFrame,text="Register",command=register,bg="white",font=("Arial", 30))
regButton.grid(row=4,column=2)

backButton = tk.Button(RegisterFrame,text="Back",command=recog_face_gender_ageFrameRaiseFrame,bg="white",font=("Arial", 30))
backButton.grid(row=4,column=4)



# tk.Label(userMenuFrame,text="Hello, ",font=("Courier",60),bg="white").grid(row=1,column=1)
# tk.Label(userMenuFrame,textvariable=loggedInUser,font=("Courier",60),bg="white",fg="red").grid(row=1,column=2)
# tk.Button(userMenuFrame,text="Back",font=("Arial", 30),command=logFrameRaiseFrame).grid(row=2,column=1)


#Load Faces
# dfu = Dlib_Face_Unlock()
raiseFrame(recog_face_gender_ageFrame)
root.mainloop()