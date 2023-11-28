import os

def replace_first_element(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        file.close()
    with open(file_path, 'w') as file:
        file.close()

    with open(file_path, 'w') as file:
        for i in range(len(lines)):
            elements = lines[i].split()
            elements[0] = '0'
            line = ' '.join(elements)
            file.write(line + '\n')

def process_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            replace_first_element(file_path)
            print(f"Processed: {filename}")

# Replace 'your_folder_path' with the path to your folder containing text files
folder_path = './norb_datasets/norb_v2/valid/labels'
process_folder(folder_path)

folder_path = './norb_datasets/norb_v2/train/labels'
process_folder(folder_path)
