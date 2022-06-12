
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

image_file = '2007_000063.jpg' # Input image in images folder
image_save = '2007_000063_kmeans.png'
k = 3 # Number of clusters
pos_factor = 100 # How important the position of pixels is; Change the range of the dimension to 1 until pos_factor

################

#working_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

image_path = os.path.join('PASCAL/', image_file)

# Read in the image
image = cv2.imread(image_path)


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


#the below line of code defines the criteria for the algorithm to stop running,
#which will happen is 100 iterations are run or the epsilon (which is the required accuracy)
#becomes 85%
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)


# then perform k-means clustering wit h number of clusters defined as 3
#also random centres are initially choosed for k-means clustering
retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# convert data into 8-bit values
centers = np.uint8(centers)
segmented_data = centers[labels.flatten()]

# reshape data into the original image dimensions
segmented_image = segmented_data[:,:3].reshape((image_Lab.shape))


image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# plt.figure('Original Image')
# plt.imshow(image)

plt.figure('Segmented Image')
plt.imshow(segmented_image)

plt.imsave(image_save, segmented_image)

plt.show()

