import pandas as pd
import shutil

# Path to the Excel file
excel_file_path = "path/to/your/excel_file.xlsx"

# Original folder path
original_folder = "path/to/your/original_folder/"

# Destination folder path
destination_folder = "path/to/your/destination_folder/"

# Read the Excel file into a DataFrame
df = pd.read_excel(excel_file_path)

# Iterate through each row
for index, row in df.iterrows():
    image_name = row['image_name']
    probability = row['probability']

    # Check if probability is greater than 0.9
    if probability > 0.9:
        # Path to the original image
        original_image_path = original_folder + image_name

        # Move the image to the destination folder
        shutil.move(original_image_path, destination_folder)
        print(f"Moved {image_name} to {destination_folder}")
        
        
#########################################################################################################################################
import os
import pandas as pd
import shutil

# Path to the Excel file
excel_file_path = "path/to/your/excel_file.xlsx"

# Original folder path
original_folder = "path/to/your/original_folder/"

# Destination folder path
destination_folder = "path/to/your/destination_folder/"

# Read the Excel file into a DataFrame
df = pd.read_excel(excel_file_path)

# Iterate through each row
for index, row in df.iterrows():
    image_name = row['image_name']
    probability = row['probability']

    # Check if probability is greater than 0.9
    if probability > 0.9:
        # Path to the original image
        original_image_path = original_folder + image_name

        # Check if the image exists in the original folder
        if os.path.exists(original_image_path):
            # Move the image to the destination folder
            shutil.move(original_image_path, destination_folder)
            print(f"Moved {image_name} to {destination_folder}")
        else:
            print(f"Image {image_name} not found in the original folder.")
    else:
        print(f"Image {image_name} does not meet the probability threshold.")

#########################################################################################################################################



