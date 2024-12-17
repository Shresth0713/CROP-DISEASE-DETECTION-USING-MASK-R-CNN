from PIL import Image
import os

input_dir = "D:\\minor\\test"
output_dir = "D:\\minor\\test_resized"
target_size = (1024, 1024) # specify the desired size in pixels

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for filename in os.listdir(input_dir): # change the extensions as needed
    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename)
    with Image.open(input_path) as img:
        img = img.convert('RGB')
        img_resized = img.resize(target_size)
        img_resized.save(output_path)
