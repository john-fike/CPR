from ultralytics import YOLO
from PIL import Image
import numpy as np

# Load a pretrained YOLOv8n model
model = YOLO('./runs/detect/train5/weights/best.pt')

# Run inference on an image
results = model('/home/a/Documents/VS/CPR/recolonyimages')

# View results
for r in results:
    im_array = r.plot(conf=False, probs=False, labels=False, boxes=False)  # plot a BGR numpy array of predictions
    
    # Create a copy of the original image for each condition
    green_im_array = im_array.copy()
    red_im_array = im_array.copy()

    # Check the center of each predicted box and change box color accordingly
    for box in r.boxes:
        boxCenter = (box.xywhn[0, 0].item(), box.xywhn[0, 1].item())
        if (0.1 <= boxCenter[0] <= 0.9) and (0.1 <= boxCenter[1] <= 0.9):
            green_im_array = r.plot(conf=False, probs=False, labels=False, colors=[(0, 255, 0)], img=green_im_array)  # Green color
        # else:
            # red_im_array = r.plot(conf=False, probs=False, labels=False, colors=[(255, 0, 0)], img=red_im_array)  # Red color

    # Convert the modified arrays to PIL images
    im = Image.fromarray(im_array[..., ::-1])
    green_im = Image.fromarray(green_im_array[..., ::-1])  # RGB PIL image
    red_im = Image.fromarray(red_im_array[..., ::-1])  # RGB PIL image

    # Display and save the images
    green_im.show()  # show image with green boxes
    red_im.show()  # show image with red boxes
    green_im.save('green_results.jpg')  # save image with green boxes
    red_im.save('red_results.jpg')  # save image with red boxes

    input("")
