from PIL import Image # pip install Pillow
import sys
import glob
from PIL import ImageOps
import numpy as np


# Trim all png images with white background in a folder
# Usage "python PNGWhiteTrim.py ../someFolder padding"



try:
    folderName = sys.argv[1]
    padding = int(sys.argv[2])
    padding = np.asarray([-1*padding, -1*padding, padding, padding])
except :
    print("Usage: python PNGWhiteTrim.py ../someFolder padding")
    sys.exit(1)

filePaths = glob.glob(folderName + "/*.png") #search for all png images in the folder

for filePath in filePaths:
    image=Image.open(filePath)
    image.load()
    imageSize = image.size

    # remove alpha channel
    invert_im = image.convert("RGB")

    # invert image (so that white is 0)
    invert_im = ImageOps.invert(invert_im)
    imageBox = invert_im.getbbox()
    imageBox = tuple(np.asarray(imageBox)+padding)

    cropped=image.crop(imageBox)
    print(filePath, "Size:", imageSize, "New Size:", imageBox)
    cropped.save(filePath)
