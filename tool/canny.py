import cv2
import matplotlib.pyplot as plt
img = cv2.imread("croped.png")

edge_img = cv2.Canny(img,160,80)
# cv2.imshow("Detected Edges", edge_img)
cv2.imwrite("edge.png",edge_img)
plt.imshow(edge_img)
plt.show()
# cv2.imwrite("xuehua_cany.png",edge_img)