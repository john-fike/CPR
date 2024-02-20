import CPR_tools as cpr
import os
from gtts import gTTS
if __name__ == "__main__":
    
    # DISPLAY_IMAGE_PATH ='C:/Users/John Fike/OneDrive/Documents/Visual Studio 2022/CPR/images/realTest_v3/unprocessed/WIN_20231113_12_53_28_Pro.jpg'
    # PREDICTION_FILE_PATH ='C:/Users/John Fike/OneDrive/Documents/Visual Studio 2022/CPR/runs/detect/predict5/labels/WIN_20231113_12_53_28_Pro.txt'
    # PREDICTION_FILE_PATH = 'C:/Users/John Fike/OneDrive/Documents/Visual Studio 2022/CPR/runs/detect/predict/labels/WIN_20231113_12_53_28_Pro.txt'
    # DISPLAY_TIME = 5000
    # cpr.discriminate(PREDICTION_FILE_PATH, DISPLAY_IMAGE_PATH, DISPLAY_TIME)
    # cpr.showPredictions(good_colony_file_path='C:/Users/John Fike/OneDrive/Documents/Visual Studio 2022/CPR/output/good_colonies/WIN_20231113_12_53_28_Pro.txt', 
    #                     bad_colony_file_path= 'C:/Users/John Fike/OneDrive/Documents/Visual Studio 2022/CPR/output/bad_colonies/WIN_20231113_12_53_28_Pro.txt', image_path=DISPLAY_IMAGE_PATH,
                        # display_time=DISPLAY_TIME)
    # cpr.showColonies('C:/Users/John Fike/OneDrive/Documents/Visual Studio 2022/CPR/output/good_colonies/WIN_20231113_12_53_28_Pro.txt', DISPLAY_IMAGE_PATH, margin=2)
    # showPredictions(good_colony_file_path= './output/hough/WIN_20231113_12_53_28_Pro.txt', image_path = 'C:/Users/John Fike/OneDrive/Documents/Visual Studio 2022/CPR/images/realTest_v3/unprocessed/WIN_20231113_12_53_28_Pro.jpg')

    # prediction_folder_path = 'C:/Users/John Fike/OneDrive/Documents/Visual Studio 2022/CPR/output/hough/'
    prediction_folder_path = 'C:/Users/John Fike/OneDrive/Documents/Visual Studio 2022/CPR/runs/detect/predict22/labels/'
    display_image_folder_path = 'C:/Users/John Fike/OneDrive/Documents/Visual Studio 2022/CPR/images/realTest_v3/unprocessed/'

    good_colony_folder_path = 'C:/Users/John Fike/OneDrive/Documents/Visual Studio 2022/CPR/output/good_colonies/'
    bad_colony_folder_path = 'C:/Users/John Fike/OneDrive/Documents/Visual Studio 2022/CPR/output/bad_colonies/'
    
    binary_save_folder_path = 'C:/Users/John Fike/OneDrive/Documents/Visual Studio 2022/CPR/output/binary/'
    # binary_save_folder_path = None

    predictions_save_folder_path = 'C:/Users/John Fike/OneDrive/Documents/Visual Studio 2022/CPR/output/predictions/'
    # predictions_save_folder_path = None

    display_time = 1000

    # for image in os.listdir(display_image_folder_path):

    image = os.listdir(display_image_folder_path)[0]        #!!!!!!!!!TEST!!!!!!!!!
    # find image name,
    file_name = os.path.splitext(os.path.basename(image))[0]
    # find prediction file path
    prediction_file_path = os.path.join(prediction_folder_path, file_name + '.txt')
    image_file_path = os.path.join(display_image_folder_path, image)

    good_colony_file_path = os.path.join(good_colony_folder_path, file_name + '.txt')
    bad_colony_file_path =  os.path.join(bad_colony_folder_path, file_name + '.txt')

    process_petri_dish_image('./images/realTest_v3/processed')

    # cpr.discriminate(prediction_file_path, image_file_path, binary_bad_display=False, binary_good_display=False, display_time=3030, binary_discrimination_margin = 1, good_output_path=good_colony_folder_path, bad_output_path=bad_colony_folder_path, binary_save_folder_path=binary_save_folder_path)
    # cpr.showPredictions(good_colony_file_path= good_colony_file_path,  
                        # bad_colony_file_path=bad_colony_file_path,
                        # image_path=image_file_path,
                        # display_time=10000,
                        # save_folder_path=predictions_save_folder_path)
    # cpr.showColonies(good_colony_file_path, image_file_path, display_time=10, margin=2)


    # if binary_save_folder_path is not None:
    #     text = " . I'm Done!"
    #     language = 'en'
    #     speech = gTTS(text=text, lang=language, slow=False)
    #     speech.save("output.mp3")
    #     os.system("start output.mp3")  # For Windows
        
    # cv2.waitKey(0)

    # binary_save_folder_path_good = os.path.join(binary_save_folder_path, 'good_colonies')
    # binary_save_folder_path_bad = os.path.join(binary_save_folder_path, 'bad_colonies')

    # for file in os.listdir(binary_save_folder_path_good):
    #     os.remove(os.path.join(binary_save_folder_path_good, file))

    # for file in os.listdir(binary_save_folder_path_bad):
    #     os.remove(os.path.join(binary_save_folder_path_bad, file))
    






