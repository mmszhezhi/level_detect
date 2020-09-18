from tool.utils import *
import sys
from keras_segmentation.pretrained import pspnet_101_voc12

if __name__ == '__main__':

    src = sys.argv[1]
    masks = sys.argv[2]
    dst = sys.argv[3]
    # src,masks,dst =  "beerimgs", "beermasked" ,"croped"
    assert src and dst and masks ,"src and dst needed!"
    crop_masked_imgs(src,masks,dst)