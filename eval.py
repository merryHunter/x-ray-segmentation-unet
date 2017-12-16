import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import params
import matplotlib.pyplot as plt
import PIL
import os
import eval_segm as metric

input_size = params.input_size
batch_size = params.batch_size
orig_width = params.orig_width
orig_height = params.orig_height
threshold = params.threshold
model = params.model_factory()

df_test = None #pd.read_csv('input/sample_submission.csv')
ids_test = None #df_test['img'].map(lambda s: s.split('.')[0])

OUTPUT_DIR = 'output/'
N_CLASSES = 2

def get_test_mask_paths(filename):
    path_set = []
    with open(filename, "r") as f:
        lines = f.readlines()
        for l in lines:
            if l != '\n':
                l = l.split(' ')
                path_set.append((l[0],l[1].strip()))
                im = cv2.imread(l[1],cv2.IMREAD_GRAYSCALE )

    return path_set

names = get_test_mask_paths("train.txt")
# for id in ids_test:
#     names.append('{}.jpg'.format(id))


# https://www.kaggle.com/stainsby/fast-tested-rle
def run_length_encode(mask):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    inds = mask.flatten()
    runs = np.where(inds[1:] != inds[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    rle = ' '.join([str(r) for r in runs])
    return rle


rles = []

model.load_weights(filepath='weights/best_weights.hdf5')


def read_mask_image(mask_img):
    mask_img[mask_img == False] = 0
    mask_img[mask_img == True] = 1
    return mask_img

print('Predicting on {} samples with batch_size = {}...'.format(len(names), batch_size))


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

pixel_accuracy = []


def get_pred_orig_grayscale(pred, orig):
    image_pred = np.zeros((orig_height, orig_width), dtype="uint8")
    image_pred[np.where((pred != [0, 0, 0]).all(axis=2))] = np.ones((1, 1), dtype="uint8")

    # select indexes where green in intensive so it's a mask
    mask = orig[:,:,1] > 230
    image_orig = np.zeros((orig_height, orig_width, 3), dtype="uint8")
    image_orig[mask] = np.ones((1, 1, 1), dtype="uint8")

    # get back to 2-d array
    image_orig = image_orig[:,:,0]

    return image_pred, image_orig


for start in tqdm(range(0, len(names), batch_size)):
    x_batch = []
    end = min(start + batch_size, len(names))
    ids_test_batch = names[start:end]
    ns = []
    for name in ids_test_batch:
        ns.append((name[0],name[1]))
        img = cv2.imread(name[0])
        img = cv2.resize(img, (input_size, input_size))
        x_batch.append(img)
    x_batch = np.array(x_batch, np.float32) / 255
    preds = model.predict_on_batch(x_batch)
    preds = np.squeeze(preds, axis=3)
    plt.figure(figsize=(20, 20))
    i = 0

    for pred in preds:
        print ns[i][0]
        img_name = (ns[i][0].split('/')[-1]).split('.')[0] + '/'
        cur_dir = OUTPUT_DIR + img_name
        os.mkdir(OUTPUT_DIR + img_name)
        prob = cv2.resize(pred, (orig_width, orig_height))
        orig_img = cv2.imread(ns[i][0])
        cv2.imwrite('temp.jpg', prob)
        p = cv2.imread('temp.jpg')
        mask_orig = cv2.imread(ns[i][1])

        # visualize(orig_img, 255 * prob, mask_orig)

        # get and write transparent mask for predicted image
        output_pred = get_transparent_prediction(orig_img, p,alpha=0.5,orig=False)
        cv2.imwrite(cur_dir + str(i) + '_' + ns[i][1].split('/')[-1], output_pred)

        # get and write transparent mask for ground truth
        output = get_transparent_prediction(orig_img, mask_orig,alpha=0.5,orig=True)
        cv2.imwrite(cur_dir + 'orig_' + ns[i][1].split('/')[-1], output)
        pred, orig = get_pred_orig_grayscale(p, mask_orig)
        cv2.imwrite(cur_dir + 'mask_pred_' + ns[i][1].split('/')[-1],255 * pred)
        cv2.imwrite(cur_dir + 'mask_orig_' + ns[i][1].split('/')[-1],255 * orig)
        iou = metric.pixel_accuracy(pred, orig)
        if iou != -1:
            pixel_accuracy.append(iou)
        print "MEAN accuracy: {0}".format(iou)

        # for original submission
        mask = prob > threshold
        rle = run_length_encode(mask)
        rles.append(rle)
        i += 1

print "Average pixel accuracy: {0}".format(np.sum(pixel_accuracy) / len(pixel_accuracy))
print "number of defected images: {0}".format(len(pixel_accuracy))

print("Generating submission file...")  
df = pd.DataFrame({'img': names, 'rle_mask': rles})
df.to_csv('submit/submission.csv.gz', index=False, compression='gzip')
