import os
from ultralytics import YOLO

model = YOLO('yolov8m')

input_video = 'input_videos/08fd33_4.mp4'

output_dir = 'output1'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

results = model.predict(source=input_video, save=True, save_dir=output_dir)

print(results[0])
print('=============================================================')

for box in results[0].boxes:
    print(box)
