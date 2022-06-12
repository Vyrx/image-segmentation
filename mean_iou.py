import numpy as np
import math
import cv2
from sklearn.metrics import confusion_matrix

def compute(y_pred, y_true):
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    current = confusion_matrix(y_true, y_pred)
    intersection = np.diag(current)
    ground_truth_set = current.sum(axis=1)
    predict_set = current.sum(axis=0)
    union = ground_truth_set + predict_set - intersection
    IoU = intersection / union.astype(np.float32)
    return np.mean(IoU)


if __name__ == "__main__":
    im_gt = cv2.imread("./images/2008_003709.jpg")
    im_predict = cv2.imread("./images/k_means_3.jpg")
    col, row, _ = im_gt.shape
    diff_r, diff_c = (row - 481), (col - 321)
    resize_img = np.zeros((321, 481, 3))
    a, b = 0, 0
    for i in range(math.floor(diff_c / 2.0), col - math.ceil(diff_c / 2.0), 1):
        for j in range(math.floor(diff_r / 2.0), row - math.ceil(diff_r / 2.0), 1):
            resize_img[a][b] = im_gt[i][j]
            b += 1
        b = 0
        a += 1


    label_gt = set( tuple(v) for m2d in resize_img for v in m2d )
    labels_gt = np.zeros((321, 481))
    label_pre = set( tuple(v) for m2d in im_predict for v in m2d )
    labels_pre = np.zeros((321, 481))
    for i in range(321):
        for j in range(481):
            if tuple(resize_img[i][j]) == (0, 0, 128) or tuple(resize_img[i][j])  == (0, 128, 0):
                labels_gt[i][j] = 1
            if tuple(im_predict[i][j]) == (144, 144, 80):
                labels_pre[i][j] = 1
            if tuple(im_predict[i][j]) == (218, 129, 120):
                labels_pre[i][j] = 2
            if tuple(im_predict[i][j]) == (204, 81, 33):
                labels_pre[i][j] = 0
    ans = compute(labels_pre, labels_gt)
    print(ans)



"""
x = np.zeros((50, 50, 3))
for i in range(50):
    for j in range(50):
        # (144, 144, 80) : foreground
        # (218, 129, 120) : light
        # (204, 81, 33) : background
        x[i][j] = (0.0, 0.0, 128.0)
cv2.imwrite( "test.png", x )
"""