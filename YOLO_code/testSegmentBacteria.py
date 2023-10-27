from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO('yolov8n-seg.pt')

# Run inference on 'bus.jpg' with arguments
model.predict('sethTest.jpg', save=True, save_txt = True, classes = None, agnostic_nms = True, hide_labels = True, hide_conf = True)
