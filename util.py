import cv2
import numpy as np
import os
import eval_segm as metric
import matplotlib.pyplot as plt


def visualize(img_orig, mask_pred, mask_orig):
    mask_pred = get_transparent_prediction(img_orig, mask_pred)
    plt.subplot(131)
    plt.imshow(img_orig)
    plt.subplot(132)
    plt.imshow(mask_orig)
    plt.subplot(133)
    plt.imshow(mask_pred)
    plt.show()


def get_transparent_prediction(img_orig, mask_pred, alpha=0.5, orig=True):
    output = img_orig.copy()
    if orig:
        image = mask_pred
    else:
        image = np.zeros((orig_height, orig_width, 3), dtype="uint8")
        image[np.where((mask_pred != [0, 0, 0]).all(axis=2))] = [255, 0, 0]

    overlay = image.copy()
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    return output


def get_pred_orig_grayscale(pred, orig):
    image_pred = np.zeros((orig_height, orig_width), dtype="uint8")
    image_pred[np.where((pred != [0, 0, 0]).all(axis=2))] = np.ones((1, 1), dtype="uint8")

    # select indexes where green in intensive so it's a mask
    mask = orig[:, :, 1] > 230
    image_orig = np.zeros((orig_height, orig_width, 3), dtype="uint8")
    image_orig[mask] = np.ones((1, 1, 1), dtype="uint8")

    # get back to 2-d array
    image_orig = image_orig[:, :, 0]

    return image_pred, image_orig


def get_iou(prediction, ns, i, write_pred=True):
    print ns[i][0]
    img_name = (ns[i][0].split('/')[-1]).split('.')[0] + '/'
    cur_dir = OUTPUT_DIR + img_name
    os.mkdir(OUTPUT_DIR + img_name)
    prob = cv2.resize(prediction, (orig_width, orig_height))
    orig_img = cv2.imread(ns[i][0])
    cv2.imwrite('temp.jpg', prob)
    p = cv2.imread('temp.jpg')
    mask_orig = cv2.imread(ns[i][1])
    # visualize(orig_img, 255 * prob, mask_orig)
    pred, orig = get_pred_orig_grayscale(p, mask_orig)

    if write_pred:
        # get and write transparent mask for predicted image
        output_pred = get_transparent_prediction(orig_img, p, alpha=0.5, orig=False)
        cv2.imwrite(cur_dir + str(i) + '_' + ns[i][1].split('/')[-1], output_pred)

        # get and write transparent mask for ground truth
        output = get_transparent_prediction(orig_img, mask_orig, alpha=0.5, orig=True)
        cv2.imwrite(cur_dir + 'orig_' + ns[i][1].split('/')[-1], output)
        cv2.imwrite(cur_dir + 'mask_pred_' + ns[i][1].split('/')[-1], 255 * pred)
        cv2.imwrite(cur_dir + 'mask_orig_' + ns[i][1].split('/')[-1], 255 * orig)

    return metric.pixel_accuracy(pred, orig)
