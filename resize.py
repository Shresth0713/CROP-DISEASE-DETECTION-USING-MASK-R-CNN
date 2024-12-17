import os
import json
from PIL import Image

# Set the desired size for the images
new_size = (1024, 1024)

# Set the path to your dataset directory
dataset_dir = "D:\\rice_diseases-20230504T104336Z-001\\New folder\\val"
dataset_dir2 = "D:\\minor\\rice\\val"
annotations = json.load(open('D:\\minor\\labels\\rice_test4.json'))
label=[]
# Loop through each file in the dataset directory
for a in annotations:
    print(a)
    filename=a['filename']
        # Load the image using PIL
    print(filename)
    img = Image.open(os.path.join(dataset_dir, filename))
    img = img.convert('RGB')
        # Resize the image
    img_resized = img.resize(new_size)

        # Save the resized image
    img_resized.save(os.path.join(dataset_dir2, filename))

        # Open the corresponding JSON file and update the label information


        # Update the size of the image

        # Loop through each region and update the coordinates
    for region in a["regions"]:
        for i in range(len(region["shape_attributes"]["all_points_x"])):
            region["shape_attributes"]["all_points_x"][i] = int(
                region["shape_attributes"]["all_points_x"][i] * new_size[0] / img.size[0])
            region["shape_attributes"]["all_points_y"][i] = int(
                region["shape_attributes"]["all_points_y"][i] * new_size[1] / img.size[1])
    label.append(a)
        # Save the updated label information

with open(os.path.join(dataset_dir2, "rice.json"), "w") as f:
    json.dump(label, f)