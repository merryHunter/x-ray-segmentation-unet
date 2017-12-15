import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import params
import matplotlib.pyplot as plt

input_size = params.input_size
batch_size = params.batch_size
orig_width = params.orig_width
orig_height = params.orig_height
threshold = params.threshold
model = params.model_factory()

df_test = None #pd.read_csv('input/sample_submission.csv')
ids_test = None #df_test['img'].map(lambda s: s.split('.')[0])

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

names = get_test_mask_paths("val.txt")
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

print('Predicting on {} samples with batch_size = {}...'.format(len(names), batch_size))
for start in tqdm(range(0, len(names), batch_size)):
    x_batch = []
    end = min(start + batch_size, len(names))
    ids_test_batch = names[start:end]
    for name in ids_test_batch:
        img = cv2.imread(name[0])
        img = cv2.resize(img, (input_size, input_size))
        x_batch.append(img)
    x_batch = np.array(x_batch, np.float32) / 255
    preds = model.predict_on_batch(x_batch)
    preds = np.squeeze(preds, axis=3)
    for pred in preds:
        prob = cv2.resize(pred, (orig_width, orig_height))
        mask = prob > threshold
        print mask

        rle = run_length_encode(mask)
        print "rle"
        print rle
        cv2.imwrite("im.jpg", rle)
        rles.append(rle)

print("Generating submission file...")
df = pd.DataFrame({'img': names, 'rle_mask': rles})
df.to_csv('submit/submission.csv.gz', index=False, compression='gzip')
