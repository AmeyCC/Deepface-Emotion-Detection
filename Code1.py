import cv2
from deepface import DeepFace
cap=cv2.VideoCapture("Video 1.mp4")
output = cv2.VideoWriter("Output1.mp4",cv2.VideoWriter_fourcc(*'mp4v'),24,(1280,720))
while cap.isOpened():
    _,frame=cap.read()
    if _ != True:
        break

    try:
        predictions = DeepFace.analyze(frame, actions=['emotion'])#Predict emotions for each frame
        emotion = predictions['dominant_emotion'] # Get the dominant emotion
        cv2.putText(frame,emotion,(5,20),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2) # Write the dominant emotion
        x,y,w,h=int(predictions['region']['x']),int(predictions['region']['y']),int(predictions['region']['w']),int(predictions['region']['h'])
        cv2.rectangle(frame, (x, y), (x + w, y + w), (0, 255, 0), 1)
    except Exception:
        print("No Face Available")
    frame = cv2.resize(frame, (1280, 720))
    cv2.imshow("video", frame)
       #result.write(frame)

    output.write(frame)
    key = cv2.waitKey(1) & 0xFF
    if key==ord('q'):
        break
cap.release()
output.release()
cv2.destroyAllWindows()