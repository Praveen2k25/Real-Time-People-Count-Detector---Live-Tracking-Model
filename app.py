import cv2
import torch

model = torch.hub.load('ultralytics/yolov5', 'yolov5s') 

cap = cv2.VideoCapture(0)  

while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame)
    detections = results.pandas().xyxy[0] 
    person_detections = detections[detections['name'] == 'person']

 
    for index, row in person_detections.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)


    cv2.imshow('Person Detection', frame)


    person_count = len(person_detections)
    print(f'Number of people detected: {person_count}')

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
