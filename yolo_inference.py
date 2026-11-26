from ultralytics import YOLO

model = YOLO('models/court/best.pt')  # load a manually trained model

result = model.predict('input_videos/fr_jp_fragment.mp4', save=True)
print(result)