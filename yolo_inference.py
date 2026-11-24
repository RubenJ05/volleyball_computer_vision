from ultralytics import YOLO

model = YOLO('models/last.pt')  # load a pretrained YOLOv8n model (nano model)

result = model.predict('input_videos/fr_jp_fragment.mp4', save=True)
print(result)