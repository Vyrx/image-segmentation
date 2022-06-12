import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import os

image_file = 'geo.jpeg' # Input image in images folder
image_save = 'out.jpg'
par_eps = 5
par_sam = 50

#############################

image_path = os.path.join('images/', image_file)

image= cv2.imread(image_path) 

pixel_vals = image.reshape((-1,3))
 
# Convert to float type
pixel_vals = np.float32(pixel_vals)

db = DBSCAN(eps=par_eps, min_samples=par_sam).fit(pixel_vals)

segmented_image = np.uint8(db.labels_.reshape(image.shape[:2]))

# plt.figure('Original Image')
# plt.imshow(image)

plt.figure('Segmented Image')
plt.imshow(segmented_image)

plt.imsave(image_save, segmented_image)

plt.show()

