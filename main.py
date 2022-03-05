import cv2
from tracker import*
cap=cv2.VideoCapture("vid.mp4")
cap.set(1,450)
#success,frame=cap.read()
#frame=cv2.resize(frame,(640,480))
#roi=cv2.selectROI(frame)
71,208,506,330
obj=cv2.createBackgroundSubtractorMOG2(history=100,varThreshold=70)
tracker=Tracker()
while True:
    success,frame=cap.read()
    frame=cv2.resize(frame,(640,480))
    roi=frame[208:330,71:506]
    mask=obj.apply(roi)
    _,mask=cv2.threshold(mask,254,255,cv2.THRESH_BINARY)
    cnt,_=cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    points=[]
    for c in cnt:
        area=cv2.contourArea(c)
        if area > 400:
#            cv2.drawContours(roi,[c],-1,(0,255,0),2)
           x,y,w,h=cv2.boundingRect(c)
           points.append([x,y,w,h])
    point=tracker.update(points)
    for i in point:
        x,y,w,h,id=i
        cv2.rectangle(roi,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.putText(roi,str(id),(x,y -1),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
           
    cv2.imshow("FRAME",frame)
    if cv2.waitKey(32)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()