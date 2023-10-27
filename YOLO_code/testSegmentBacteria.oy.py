from ultralytics import SAM
import cv2

model = SAM('sam_b.pt')
results = model("testFuckShitAss.jpg")
annotated_image = results[0].plot()


# Check if the image was loaded successfully
if annotated_image is not None:
    # Display the image
    cv2.imshow('Image', annotated_image)
    #wait for keystroke to close window
    cv2.waitKey(0) 
else:
    print("Failed to load the image or the file does not exist.")
