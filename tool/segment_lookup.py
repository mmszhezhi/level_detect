import numpy as np
import cv2,os
from tool import utils
# raw = cv2.imread("../beer.png")
# seg = cv2.imread("../beer_seg.png")
# bins = np.sum(seg,axis=2)
# bins = np.clip(bins<300,0,1)
# p= bins.dot(raw)
# utils.plot(p)
# utils.visualize([p])
# bins.shape

def add_mask2image_binary(images_path, masks_path, masked_path):
# Add binary masks to images
    for img_item in os.listdir(images_path):
        print(img_item)
        img_path = os.path.join(images_path, img_item)
        img = cv2.imread(img_path)
        mask_path = os.path.join(masks_path, img_item[:-4]+'.png')
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        masked = cv2.add(img, np.zeros(np.shape(img), dtype=np.uint8), mask=mask)
        cv2.imwrite(os.path.join(masked_path, img_item), masked)


def extract(images_path, masks_path, masked_path):
    # Add binary masks to images
    for img_item in images_path:
        print(img_item)
        img = cv2.imread(img_item)
        mask = cv2.imread(masks_path,cv2.IMREAD_GRAYSCALE)
        # mask = np.clip(mask>150,0,255)
        masked = cv2.bitwise_not(img,np.zeros(np.shape(img),dtype=np.uint8),mask=mask)

        # masked = cv2.add(img, np.zeros(np.shape(img), dtype=np.uint8), mask=mask)
        cv2.imwrite(masked_path,masked)


images_path = '../beer.png'
masks_path = '../beer_seg.png'
masked_path = 'test.png'
# extract([images_path], masks_path, masked_path)


def mask_img(img,mask,masked,crop=False):
    img = cv2.imread(img)
    label = cv2.imread(mask)
    label = cv2.cvtColor(label,cv2.COLOR_BGR2GRAY)
    # label = np.clip(label>130,0,255)
    label[label<130] = 255
    label[label < 200] = 0
    masked_img = cv2.bitwise_and(img, img, mask=label)
    if crop:
        x,y,w,h=cv2.boundingRect(cv2.cvtColor(masked_img,cv2.COLOR_BGR2GRAY))
        # cv2.rectangle(masked_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        croped = cv2.getRectSubPix(masked_img,(w,h),((x+w/2),(y+h/2)))
    cv2.imwrite(masked, masked_img)
    cv2.imwrite("croped.png", croped)


mask_img("../xuehua.jpg","../xuehua_mask.png","xuehua_crop.png",True)
