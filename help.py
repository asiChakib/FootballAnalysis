import os
from ultralytics import YOLO

# Load the YOLO model
model = YOLO('models/best.pt')

# Define input video and output directory
input_video = 'input_videos/08fd33_4.mp4'
output_dir = 'output1'

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Perform prediction and save the results
# Explicitly specify save_dir and save=True to ensure results are saved in the correct directory
results = model.predict(source=input_video, save=True, project=output_dir, name='', exist_ok=True)

# Print the first result and its boxes
print(results[0])
print('=============================================================')

# Iterate over the detection boxes and print them
for box in results[0].boxes:
    print(box)
    print('=============================================================')
