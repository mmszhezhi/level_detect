import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
from keras_segmentation.pretrained import pspnet_50_ADE_20K , pspnet_101_cityscapes, pspnet_101_voc12
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
from keras_segmentation.predict import predict_multiple
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# model = pspnet_50_ADE_20K() # load the pretrained model trained on ADE20k dataset
# model = pspnet_101_cityscapes() # load the pretrained model trained on Cityscapes dataset
model = pspnet_101_voc12() # load the pretrained model trained on Pascal VOC 2012 dataset
out = model.predict_segmentation(
    inp="xuehua1.png",
    out_fname="xuehua_mask1.png"
)

def segment(path,out):
    out = model.predict_segmentation(inp=path,out_fname=out)
    return out


# predict_multiple(
#     model=model,
# 	inp_dir="testdir",
# 	out_dir="testresult"
# )