import os
import cv2

with open("train.txt","w") as out:
  base = "/home/ivan/projects/internship_addfor/nets/KittiSeg/data_tooth/"
  for f in os.listdir("data_tooth/"):
    if "_mask" not in f:
      out.write(base + f + " " + base + f.split('.')[0] + '_mask.jpg')
      out.write('\n')
