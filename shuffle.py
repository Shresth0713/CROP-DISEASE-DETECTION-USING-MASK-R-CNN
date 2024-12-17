import json
import random
annotations1 = json.load(open("D:\\minor\\potato2\\train\\potato.json"))
random.shuffle(annotations1)
print(annotations1)