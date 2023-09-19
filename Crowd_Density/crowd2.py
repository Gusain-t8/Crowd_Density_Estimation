import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import filedialog
import time
from scipy.spatial import distance as distance
import cmath



#creating window using tkinter 
root = Tk()
root.title(" People Density Estimation ")
root.geometry("800x500")
#setting backgroud image for the window
bg = PhotoImage(file="using.png")
#blending bg image with master window
my_label=Label(root,image=bg)
my_label.place(x=0,y=0,relwidth=1,relheight=1)

#accessing classes stored in coc.names file
labelpath ='coco.names'
file = open(labelpath)
#load names of classes 
label = file.read().strip().split("\n")
label[0]

#loading pre-trained model weights and configuration
weightspath ='yolov3.weights'
configpath ='yolov3.cfg'

#loading models and its weights, stored in net obj
net = cv2.dnn.readNetFromDarknet(configpath, weightspath)

#using net obj, getLayesNames return list of string,
#where each string is name of layer in the network
layer_names = net.getLayerNames()
#list of name of unconnected output layer  of net obj
ln = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]


def videocheck():
    i=0
    #dialog box to select video file
    fln=filedialog.askopenfilename(initialdir=os.getcwd(),title="Open file",filetypes=(("MP4","*.mp4"),("All File","*.*")))
    videopath =fln
    # define a video capture object
    video = cv2.VideoCapture(videopath)
    ret = video


    #infinite loop to read the frames using video object
    data=[]
    while(True):
        #capturing the video frame in ret, frame
        ret, frame = video.read()
        if ret == False:
            print('Error running the file :(')

        #resizing image using method cv2.INTER_AREA having pixel w=640 h=440    
        frame = cv2.resize(frame, (640, 440), interpolation=cv2.INTER_AREA)
        #input to the network blob-4d array img,channel,w,ht
        blob = cv2.dnn.blobFromImage(
            frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        r = blob[0, 0, :, :]
        #identify object andset new value to the network
        net.setInput(blob)
        t0 = time.time()
        #store detected objects consist class label,confidence score,coordinates
        outputs = net.forward(ln)
        t = time.time()

        boxes = []
        confidences = []
        classIDs = []
        center = []
        output = []
        count = 0
        results = []
        breach = set()
        #store height and  width of the image
        h, w = frame.shape[:2]
        for output in outputs:
            for detection in output:
                #extract scores - element from 5th index to last
                scores = detection[5:]
                #extract class id/max element index of scores
                classID = np.argmax(scores)
                #extract confidence score for the class
                confidence = scores[classID]

                if confidence > 0.5:
                    box = detection[0:4] * np.array([w, h, w, h])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    #updating centriod of the bounded box
                    center.append((centerX, centerY))
                    #assigning new coordinates of the box
                    box = [x, y, int(width), int(height)]
                    #append coordinates of the bounded box
                    boxes.append(box)
                    #append conf of the selected bounded box
                    confidences.append(float(confidence))
                    #append classid of selected bounded box
                    classIDs.append(classID)
        #most promising bounded boxes are stored in indices conf<0.4 iou>0.5
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        if len(indices) > 0:
            for i in indices.flatten():
                #extract cordinates(top left) x,y and wd,ht of each bounded box
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                # color = [int(c) for c in colors[classIDs[i]]]
                if(label[classIDs[i]] == 'person'):
                    # objects detected of class label are sorted here
                    cX = (int)(x+(y/2))
                    cY = (int)(w+(h/2))
                    #updating centroid cordinates
                    center.append((cX, cY))
                    res = ((x, y, x+w, y+h), center[i])
                    results.append(res)
                    #calculating Euclidean distance
                    dist = cmath.sqrt(
                        ((center[i][0]-center[i+1][0])**2)+((center[i][1]-center[i+1][1])**2))
                    #if distance between centroid is less than 100,considering it close/not safe
                    if(dist.real < 100):
                        #centroid are close color RED BGR=(0,0,255)
                        cv2.rectangle(frame, (x, y), (x+w, y+h),(0, 0, 255), 2)
                        #centroid displayed using circle
                        cv2.circle(frame, center[i], 4, (0, 0, 255), -1)
                        #distance between centriod displayed using line
                        cv2.line(frame, (center[i][0], center[i][1]), (center[i+1][0], center[i+1][1]), (0,0, 255), thickness=3, lineType=8)
                        #incrementing count of the people
                        count = count+1

                    else:
                        #centroid are far frame color GREEN BGR=(0,255,0)
                        cv2.rectangle(frame, (x, y), (x+w, y+h),(0, 255, 0), 2)
                        #diaplaying centriod 
                        cv2.circle(frame, center[i], 4, (0, 255, 0), -1)
                         #incrementing count of the people
                        count = count+1
            
           
            #displaying total count
            cv2.putText(frame, "Total Count: {}".format(
                count), (20, frame.shape[0] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
        #displaying frame with bounded boxes    
        cv2.imshow('Frame', frame)

        print(count)
        current_time =i
        #storing count of people and time in data for platting graph
        data.append((count,current_time))
        i=i+1
        #press 'q' to quit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    print(1)
    #after the looping release the object
    video.release()
    #destroy the open window
    cv2.destroyAllWindows()

    #if count> 1 display it in window
    if count >0:
        op1.delete("1.0", END)
        op1.insert(END, count)
    
    #sorting the stored data (count,time)
    def Sort(sub_li):
        sub_li.sort(key = lambda x: x[1])
        return sub_li
    
    # printing the cordinates (time,count)
    print(Sort(data))
    print(data)
    #storing count in y and time in x
    x = []
    y=[]
    for i in data:
        x.append(i[1])
    for i in data:
        y.append(i[0])
    

    #plotting the points 
    plt.plot(x, y)
    
    #labeling the x axis
    plt.xlabel('Time')
    #labeling the y axis
    plt.ylabel('Count')
    
    #title to graph
    plt.title('Density')
    
    # function to show the plot
    plt.show()

#function for counting for people
def photo():
    ret = True
    #accessing image file
    f_types= [('Image Files', '*.jpg')]
    #dialog box toopen image file
    filename = filedialog.askopenfilename(filetypes=f_types)
    #reading image file
    img=cv2.imread(filename)
    frame=img
    #displaying frame with bounded boxes
    cv2.imshow('Frame', frame)
    if ret == False:
        print('Error running the file :(')
    frame = cv2.resize(frame, (640, 440), interpolation=cv2.INTER_AREA)
    #input to the network blob-4d array img,channel,w,ht
    blob = cv2.dnn.blobFromImage(
        frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    r = blob[0, 0, :, :]
    #identify objects
    net.setInput(blob)
    t0 = time.time()
    #store detected objects consist class label,confidence score,coordinates
    outputs = net.forward(ln)
    t = time.time()

    boxes = []
    confidences = []
    classIDs = []
    center = []
    output = []
    count = 0
    results = []

    h, w = frame.shape[:2]
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)

            confidence = scores[classID]

            if confidence > 0.5:
                box = detection[0:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                center.append((centerX, centerY))
                box = [x, y, int(width), int(height)]
                boxes.append(box)
                confidences.append(float(confidence))
                classIDs.append(classID)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    if len(indices) > 0:
        for i in indices.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            # color = [int(c) for c in colors[classIDs[i]]]
            if(label[classIDs[i]] == 'person'):
                # objects detected of class label are sorted here
                cX = (int)(x+(y/2))
                cY = (int)(w+(h/2))
                #updating centroid
                center.append((cX, cY))
                res = ((x, y, x+w, y+h), center[i])
                results.append(res)
                #calculating Euclidean distance
                dist = cmath.sqrt(
                    ((center[i][0]-center[i+1][0])**2)+((center[i][1]-center[i+1][1])**2))
                #if distance between centroid is less than 100,considering it close/not safe    
                if(dist.real < 100):
                    #centroid are close color RED BGR=(0,0,255)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    #centroid displayed using circle
                    cv2.circle(frame, center[i], 4, (0, 0, 255), -1)
                    #distance between centriod displayed using line
                    cv2.line(frame, (center[i][0], center[i][1]), (center[i+1][0], center[i+1][1]), (0,0, 255), thickness=3, lineType=8)
                    #incrementing count
                    count = count+1

                else:
                    #centroid are far frame color GREEN BGR=(0,255,0)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    #diaplaying centriod
                    cv2.circle(frame, center[i], 4, (0, 255, 0), -1)
                    count = count+1
        #displaying total count
        cv2.putText(frame, "Total Count: {}".format(
            count), (20, frame.shape[0] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)

    #displaying frame with bounded boxes
    cv2.imshow('Frame', frame)

    #count > 1 display it in window
    if count >0:
        op2.delete("1.0", END)
        op2.insert(END, count)
    #destroy the image window    
    cv2.waitKey()
    cv2.destroyAllWindows()
 
#Symptom1 = StringVar() Symptom1.set(None) Symptom2 = StringVar() Symptom2.set(None) Symptom3 = StringVar() Symptom3.set(None) Symptom4 = StringVar() Symptom4.set(None) Symptom5 = StringVar() Symptom5.set(None)

#creating label for parent window
l1 = Label(root,justify=LEFT, text=" PEOPLE DENSITY ESTIMATION ")
l1.config(font=("Times New Roman", 30),background="orange")
l1.grid(row=1, column=0, columnspan=2, padx=100,pady=40)

#creating label
opt = Label(root, text="Please Select The Options  ")
opt.config(font=("Times New Roman", 20),background="orange")
opt.grid(row=5, column=0, pady=10)

#creating buttons to take input as video or image
bt = Button(root, text="VIDEO",height=2, width=10, command=videocheck)
bt.config(font=("Times New Roman", 16),background="gray")
bt.grid(row=15, column=0,pady=20)
bt = Button(root, text="IMAGE",height=2, width=10, command=photo)
bt.config(font=("Times New Roman", 16),background="gray")
bt.grid(row=16, column=0,pady=20)

#displaying count in parent window op1-video and op2-image count
op1 = Text(root, height=2, width=15)
op1.config(font=("Times New Roman", 15))
op1.grid(row=15, column=1 ,padx=60)
op2 = Text(root, height=2, width=15)
op2.config(font=("Times New Roman", 15))
op2.grid(row=16, column=1 ,padx=60)

root.mainloop()
