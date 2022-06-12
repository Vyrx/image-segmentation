import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import os

image_file = '2008_003709.jpg' # Input image in images folder
image_save = '2008_003709_dbscan.png'
par_eps = 5
par_sam = 50
pos_factor = 100 # How important the position of pixels is; Change the range of the dimension to 1 until pos_factor

#############################

image_path = os.path.join('PASCAL/', image_file)

image= cv2.imread(image_path) 

# # Change color to Lab
image_Lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

img_dimension = np.shape(image_Lab)
img_w_pos = np.zeros([img_dimension[0], img_dimension[1], 5])

for i in range(img_dimension[0]):
    for j in range(img_dimension[1]):
        img_w_pos[i][j] = np.append(image_Lab[i][j], [(i/img_dimension[0]) * pos_factor, (j/img_dimension[1]) * pos_factor])


# Reshaping the image into a 2D array of pixels and 3 color values (RGB)
pixel_vals = img_w_pos.reshape((-1,5))
 
# Convert to float type
pixel_vals = np.float32(pixel_vals)

db = DBSCAN(eps=par_eps, min_samples=par_sam).fit(pixel_vals)

print(db.labels_)

segmented_image = np.uint8(db.labels_.reshape(image.shape[:2]))

# plt.figure('Original Image')
# plt.imshow(image)

plt.figure('Segmented Image')
plt.imshow(segmented_image)

plt.imsave(image_save, segmented_image)

plt.show()

