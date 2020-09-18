from tool.utils import *
import sys
from keras_segmentation.pretrained import pspnet_101_voc12

if __name__ == '__main__':
    model = pspnet_101_voc12()
    src = sys.argv[1]
    dst = sys.argv[2]
    croped_out = sys.argv[3]
    assert src and dst ,"src and dst needed!"
    segment(model,src,dst)
    crop_masked_imgs(src, dst, croped_out)