import os
import shutil
from tqdm import tqdm

start_number = 1  # Replace with your desired start number
end_number =  12000   # Replace with your desired end number

# Source directory where images and labels are located
source_directory = "./AGAR_dataset/dataset/"

# Destination directory where you want to move image/label pairs
destination_directory = "./AGAR_dataset/use/train"

# Iterate through the range of numbers
for number in tqdm(range(start_number, end_number + 1)):
    # Construct the source file paths for image and label
    image_file = os.path.join(source_directory, f"{number}.jpg")
    label_file = os.path.join(source_directory, f"{number}.json")

    # Move the image and label files into the destination folder
    shutil.copy(image_file, os.path.join(destination_directory, f"{number}.jpg"))
    shutil.copy(label_file, os.path.join(destination_directory, f"{number}.json"))

print("Done!")