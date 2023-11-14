from ultralytics import YOLO
from ultralytics.models.sam import Predictor as SAMPredictor 
from PIL import Image
import numpy as np
import os

folder = './images/realTest_v3/processed'

# # #load model, find bacteria colonies in all images in folder path. 
# model = YOLO('AGAR_v2.pt')
# for image in os.listdir(folder):
#     model.predict(os.path.join(folder, image), save=True, imgsz=640, save_txt = True, conf = .1, classes = None, save_conf = True, hide_labels = False, hide_conf = False)  # creates 'predictions.jpg'


model = YOLO('models/norbert_v3.pt')
for image in os.listdir(folder):
    model.predict(os.path.join(folder, image), conf = .01, save=True, imgsz=640, save_txt = True, classes = None, save_conf = True, hide_labels = False, hide_conf = False)  # creates 'predictions.jpg'

# Name	Type	Default	Description
# source	str	'ultralytics/assets'	source directory for images or videos
# conf	float	0.25	object confidence threshold for detection
# iou	float	0.7	intersection over union (IoU) threshold for NMS
# imgsz	int or tuple	640	image size as scalar or (h, w) list, i.e. (640, 480)
# half	bool	False	use half precision (FP16)
# device	None or str	None	device to run on, i.e. cuda device=0/1/2/3 or device=cpu
# show	bool	False	show results if possible
# save	bool	False	save images with results
# save_txt	bool	False	save results as .txt file
# save_conf	bool	False	save results with confidence scores
# save_crop	bool	False	save cropped images with results
# hide_labels	bool	False	hide labels
# hide_conf	bool	False	hide confidence scores
# max_det	int	300	maximum number of detections per image
# vid_stride	bool	False	video frame-rate stride
# stream_buffer	bool	False	buffer all streaming frames (True) or return the most recent frame (False)
# line_width	None or int	None	The line width of the bounding boxes. If None, it is scaled to the image size.
# visualize	bool	False	visualize model features
# augment	bool	False	apply image augmentation to prediction sources
# agnostic_nms	bool	False	class-agnostic NMS
# retina_masks	bool	False	use high-resolution segmentation masks
# classes	None or list	None	filter results by class, i.e. classes=0, or classes=[0,2,3]
# boxes	bool	True	Show boxes in segmentation predictions