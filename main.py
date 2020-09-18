from tool.utils import *

img = "cola"
rawpath = f"test_cola_water/{img}.jpg"
maskpath = f"test_cola_water_mask/{img}_mask.jpg"


raw = load_img(rawpath)
mask = load_img(maskpath)
masked,croped = crop_mask_img(raw,mask,crop=True)
plt.imshow(croped)
plt.title(f"masked {img}")
plt.savefig(f"test_cola_water_mask/{img}_croped.png")
plt.clf()
sketch = canny_sketch(croped)
plt.imshow(sketch)
plt.savefig(f"test_cola_water_mask/{img}_sketch_raw.png")
plt.clf()

peaks,prop,bands,sig ,level= find_level(sketch)
plt.plot(sig)
plt.scatter(peaks[0],prop["peak_heights"][0])
plt.scatter(bands,[1,1])
plt.title(f"index of peaks {peaks[0]} \n peak height {prop['peak_heights'][0]} \n bands {bands}")
# plt.title(f"level {level} - index of peaks {peaks[0]}")

plt.savefig(f"test_cola_water_mask/{img}_signal.png")
plt.clf()
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
rect1 = plt.Rectangle((0, peaks[0]), sketch.shape[1], 0, fill=False, edgecolor = 'orange',linewidth=0.5)
ax.add_patch(rect1)
rect2 = plt.Rectangle((0, peaks[0]-int(abs(bands[0]-bands[1])) ), sketch.shape[1], 0, fill=False, edgecolor = 'orange',linewidth=0.5)
ax.add_patch(rect2)
plt.title(f"level {level} - index of peaks {peaks[0]}")
# plt.title(f"levle {peaks[0]}, bubble heights {bands}")
plt.imshow(sketch)
plt.savefig(f"test_cola_water_mask/{img}_sketch.png")
plt.show()


