import sys
from PIL import Image

if len(sys.argv) != 3:
    print("Usage: python overlay_prediction_on_image.py <image_file_path> <prediction_file_path>")

img = Image.open(sys.argv[1])
pred = Image.open(sys.argv[2])

blended = Image.blend(img, pred, 0.5)

blended.save('result.png')
