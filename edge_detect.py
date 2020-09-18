from tool import utils
from tool.canny_edge_detector import cannyEdgeDetector

imgs = utils.load_data("xuehua1.png")
# utils.visualize(imgs, 'gray')
detector = cannyEdgeDetector(imgs, sigma=1.4, kernel_size=5, lowthreshold=0.09, highthreshold=0.17, weak_pixel=100)

imgs_final = detector.detect()

utils.visualize(imgs_final, 'gray')