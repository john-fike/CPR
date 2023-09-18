from ultralytics import YOLO
from PIL import Image
# Load a pretrained YOLOv8n model
model = YOLO('./runs/detect/train5/weights/best.pt')

# Run inference on an image
results = model('/home/a/Documents/VS/CPR/recolonyimages') 

# View results
for r in results:
    im_array = r.plot(conf = False, probs = False, labels = False)  # plot a BGR numpy array of predictions
    print(r)
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    im.show()  # show image
    im.save('results.jpg')
    input("")