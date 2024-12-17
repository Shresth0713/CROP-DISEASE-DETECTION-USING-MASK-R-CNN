import cv2
import os

# Set the directory path where the images are stored
dir_path = 'D:\\minor\\potato\\train'

# Initialize variables for maximum and minimum dimensions
test_width = []
test_height = []


# Iterate through the images in the directory
for file_name in os.listdir(dir_path):
    # Load the image
    img = cv2.imread(os.path.join(dir_path, file_name))
    # Get the dimensions of the image
    height, width, _ = img.shape
    # Update the maximum and minimum dimensions
    test_width.append(width)
    test_height.append(height)

# Print the maximum and minimum dimensions
print("Maximum dimensions: {} x {}".format(max(test_width), max(test_height)))
print("Minimum dimensions: {} x {}".format(min(test_width), min(test_height)))






