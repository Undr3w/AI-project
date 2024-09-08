from ultralytics import YOLO
import streamlit as sts #for website
import cv2 #image related stuff, changing size, background ect

model = YOLO("yolov8n.pt") #YOLO class has several trained data (80) and several more feat like confidences
#open camera 
cap = cv2.VideoCapture(0)
cap.set(3, 1280) #giving dimension for the frame /width 3 and 4 are just for calling width or height
cap.set(4, 720)  #height
frame_placeholder = sts.empty()
start_button_pressed = sts.button("Start")
stop_button_pressed = sts.button("Stop")
while start_button_pressed and not stop_button_pressed:
    ret, frame = cap.read()
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    results = model(frame) #output pic
    #print(results[0].boxes)
    #break
    for result in results:
        for box in result.boxes: #boxes==>dictionary 
            x1, y1, x2, y2 = box.xyxy[0]
            conf = box.conf[0]
            class_id = int(box.cls[0]) 
            class_name = model.names[class_id] #we use this class id as key to pick out specific class from "names"
            cv2.rectangle(img_rgb, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2) #to put frame in obj
            cv2.putText(img_rgb, f'{class_name} {conf:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    frame_placeholder.image(img_rgb, channels="RGB")
while stop_button_pressed:
    break

# crt+? to comment all
#     if cv2.waitKey(1) == ord('q'):
#         break
 
# cap.release()
# cv2.destroyAllWindows()