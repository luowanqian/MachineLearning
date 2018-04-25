import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import nms


filename = 'audrey.jpg'
dets = np.array([
    [12, 84, 140, 212, 0.65],
    [24, 84, 152, 212, 0.9],
    [36, 84, 164, 212, 0.75],
    [12, 96, 140, 224, 0.8],
    [24, 96, 152, 224, 0.85],
    [24, 108, 152, 236, 0.7]])

# load image
orig_img = io.imread(filename)
nms_img = orig_img.copy()
fig, (ax1, ax2) = plt.subplots(1, 2, num='Non-maximum Suppression')

for bbox in dets:
    start_x, start_y, end_x, end_y, _ = bbox
    rect = mpatches.Rectangle((start_x, start_y),
                              end_x - start_x + 1, end_y - start_y + 1,
                              fill=False, edgecolor='red', linewidth=2)
    ax1.add_patch(rect)

# Non-maximum Suppression
picks = dets[nms.py_cpu_nms(dets, 0.3)]

for bbox in picks:
    start_x, start_y, end_x, end_y, _ = bbox
    rect = mpatches.Rectangle((start_x, start_y),
                              end_x - start_x + 1, end_y - start_y + 1,
                              fill=False, edgecolor='green', linewidth=2)
    ax2.add_patch(rect)

ax1.set_title('Before NMS')
ax1.set_axis_off()
ax1.imshow(orig_img, cmap='gray')
ax2.set_title('After NMS')
ax2.imshow(nms_img, cmap='gray')
ax2.set_axis_off()
plt.tight_layout()
plt.show()
