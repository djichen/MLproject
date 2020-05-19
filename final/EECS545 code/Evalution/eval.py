import os
import cv2
from metrics import *


y_test_folder = "groundtruth"
y_pred_folder = "agg_results"

# Run large images:
# large_true = []
# large_pred = []
# for e in os.listdir(y_test_folder):
#     if "02556" in e or "100000" in e:
#         large_true.append(e)
# for e in os.listdir(y_pred_folder):
#     if "02556" in e or "100000" in e:
#         large_pred.append(e)

# true_e = sorted(large_true)
# pred_e = sorted(large_pred)

true_e = sorted(os.listdir(y_test_folder))
pred_e = sorted(os.listdir(y_pred_folder))

pixel_acc_sum = 0
mean_pixel_acc_sum = 0
mean_iou_sum = 0
i = 0
for t in true_e:
    i += 1
    # Unet++
    # true_img = cv2.imread(os.path.join(y_test_folder, t))[2:498, 2:498, :]
    true_img = cv2.imread(os.path.join(y_test_folder, t))

    for p in pred_e:
        # if t[:-4] == p[:-12]:
        if t == p:
            print("This is the {}-th images".format(i))
            pred_img = cv2.imread(os.path.join(y_pred_folder, p))
            pixel_acc_sum += pixel_acc(true_img, pred_img)
            mean_pixel_acc_sum += mean_pixel_acc(true_img, pred_img)
            mean_iou_sum += mean_iou(true_img, pred_img)

# print("The average pixel acuracy for {}-layer Unet is {}".format(2, pixel_acc_sum / len(true_e)))
# print("The average mean pixel acuracy for {}-layer Unet is {}".format(2, mean_pixel_acc_sum / len(true_e)))
# print("The average mean IOU for {}-layer Unet is {}".format(2, mean_iou_sum / len(true_e)))

print("The average pixel acuracy for Unet++ is {}".format(pixel_acc_sum / len(true_e)))
print("The average mean pixel acuracy for Unet++ is {}".format(mean_pixel_acc_sum / len(true_e)))
print("The average mean IOU for Unet++ is {}".format(mean_iou_sum / len(true_e)))

# y_test_folder = "./your_file/"
# two_L = cv2.imread(os.path.join(y_test_folder, "2L-t18-01516_99_predict.png"))
# three_L = cv2.imread(os.path.join(y_test_folder, "3L-t18-01516_99_predict.png"))
# four_L = cv2.imread(os.path.join(y_test_folder, "4L-t18-01516_99_predict.png"))
# aggregator = cv2.imread(os.path.join(y_test_folder, "your_file.png"))
# true_img = cv2.imread(os.path.join(y_test_folder, "t18-01516_99.png"))

# print("***************************************************")
# print("The pixel acuracy for {}-layer Unet is {}".format(2, pixel_acc(true_img, two_L)))
# print("The mean pixel acuracy for {}-layer Unet is {}".format(2, mean_pixel_acc(true_img, two_L)))
# print("The mean IOU for {}-layer Unet is {}".format(2, mean_iou(true_img, two_L)))

# print("***************************************************")
# print("The pixel acuracy for {}-layer Unet is {}".format(3, pixel_acc(true_img, three_L)))
# print("The mean pixel acuracy for {}-layer Unet is {}".format(3, mean_pixel_acc(true_img, three_L)))
# print("The mean IOU for {}-layer Unet is {}".format(3, mean_iou(true_img, three_L)))

# print("***************************************************")
# print("The pixel acuracy for {}-layer Unet is {}".format(4, pixel_acc(true_img, four_L)))
# print("The mean pixel acuracy for {}-layer Unet is {}".format(4, mean_pixel_acc(true_img, four_L)))
# print("The mean IOU for {}-layer Unet is {}".format(4, mean_iou(true_img, four_L)))

# print("***************************************************")
# print("The pixel acuracy for aggregated Unet is {}".format(pixel_acc(true_img, aggregator)))
# print("The mean pixel acuracy for aggregated Unet is {}".format(mean_pixel_acc(true_img, aggregator)))
# print("The mean IOU for aggregated Unet is {}".format(mean_iou(true_img, aggregator)))