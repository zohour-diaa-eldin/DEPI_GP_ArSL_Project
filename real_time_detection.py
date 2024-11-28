import cv2
from ultralytics import YOLO

model = YOLO('ASL.pt')

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while cap.isOpened():
    
    success, frame = cap.read()

    if success:
        
        results = model.track(frame, persist=True)

        
        annotated_frame = results[0].plot()
        
        cv2.imshow("ASL Tracking", annotated_frame)


        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:

        break

cap.release()
cv2.destroyAllWindows()

