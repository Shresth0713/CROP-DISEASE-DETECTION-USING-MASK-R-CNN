# create a dictionary to group regions by filename
import json
annotations = json.load(open('D:\\minor\\labels\\rice_test3.json'))
regions_by_file = {}
for annotation in annotations:
    filename = annotation["filename"]
    if filename not in regions_by_file:
        regions_by_file[filename] = []
    regions_by_file[filename].extend(annotation["regions"])

# generate the desired output format
output = []
for filename, regions in regions_by_file.items():
    output.append({
        "filename": filename,
        "size": regions[0]["shape_attributes"].get("size", 0), # use the size from the first region, or 0 if not present
        "regions": regions
    })
# print(output)
output_file = 'D:\\minor\\labels\\rice_test4.json'
with open(output_file, 'w') as f:
    json.dump(output, f)
