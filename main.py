import cv2
from tracker import*
tracker=Tracker()
cap=cv2.VideoCapture("vid.mp4")
cap.set(1,350)
cars_cascade=cv2.CascadeClassifier('cars.xml')
#success,frame=cap.read()
#frame=cv2.resize(frame,(640,480))
#roi=cv2.selectROI(frame)
72,250,501,335
while True:
    success,frame=cap.read()
    frame=cv2.resize(frame,(640,480))
    roi=frame[250:335,72:501]
    gray=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    cars=cars_cascade.detectMultiScale(gray,1.1,1)
    detections=[]
    for (x,y,w,h) in cars:
        detections.append([x,y,w,h])
    detect=tracker.update(detections)
    for i in detect:
        x,y,w,h,id=i
        cv2.rectangle(roi,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.putText(roi,str(id),(x,y -1),cv2.FONT_HERSHEY_COMPLEX,1,(255,0, 0),2)
        
    cv2.imshow("Frame",frame)
    if cv2.waitKey(25)&0xFF==27:
        break
cap.release()
cv2.destroyAllwindows()