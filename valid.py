from ultralytics import YOLO

model = YOLO('./runs/detect/train5/weights/best.pt')

metrics = model.val()
metrics.box.map
metrics.box.map50
metrics.box.map75
metrics.box.maps