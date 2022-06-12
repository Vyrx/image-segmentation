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
    # ground truth is .png not .jpg
    im_gt = cv2.imread("./PASCAL/2008_003709.png")
    im_predict = cv2.imread("./PASCAL/2008_003709_dbscan.png")
    
    # yours should not use this part
    """
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
    """

    label_gt = set( tuple(v) for m2d in im_gt for v in m2d )
    labels_gt = np.zeros((333, 500))
    label_pre = set( tuple(v) for m2d in im_predict for v in m2d )
    print(label_pre)
    labels_pre = np.zeros((333, 500))
    
    # the object in ground truth image maybe different color with the predicted image
    for i in range(321):
        for j in range(481):
            if tuple(im_gt[i][j]) == (0, 0, 128) or tuple(im_gt[i][j])  == (0, 128, 0):
                labels_gt[i][j] = 1
            if tuple(im_predict[i][j]) == (101, 18, 71) or tuple(im_predict[i][j]) == (94, 11, 70):
                labels_pre[i][j] = 1
            if tuple(im_predict[i][j]) == (36, 231, 253):
                labels_pre[i][j] = 2
            if tuple(im_predict[i][j]) == (84, 1, 68):
                labels_pre[i][j] = 0
    ans = compute(labels_pre, labels_gt)
    print(ans)


# check the rgb is which color 

# x = np.zeros((50, 50, 3))
# for i in range(50):
#     for j in range(50):
#         # (144, 144, 80) : foreground
#         # (218, 129, 120) : light
#         # (204, 81, 33) : background
#         x[i][j] = (140, 136, 185)
# cv2.imwrite( "test.png", x )

