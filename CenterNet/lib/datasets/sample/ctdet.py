import torch.utils.data as data
import numpy as np
import torch
import json
import cv2
import os
import math

class CTDetDataset(data.Dataset):
    def _coco_box_to_bbox(self,box):
        bbox = np.array([box[0],box[1],box[0]+box[2],box[1]+box[3]],
                        dtype=np.float32)
        return bbox

    def _get_border(self,border,size):
        i=1
        while size-border // i<=border // i:
            i*=2
        return border//i

    def __getitem__(self, index):
        img_id = self.images[index]
        file_name = self.coco.loadImgs(ids=[img_id])
        img_path = os.path.join(self.img_dir,file_name)
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ids=ann_ids)
        num_objs = min(len(anns), self.max_objs)

        img = cv2.imread(img_path)

        height,width = img.shape[0],img.shape[1]
        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
        if self.opt.keep_res:
            input_h = (height | self.opt.pad) + 1
            input_w = (width | self.opt.pad) + 1
            s = np.array([input_w, input_h], dtype=np.float32)
        else:
            s = max(img.shape[0], img.shape[1]) * 1.0
            input_h, input_w = self.opt.input_h, self.opt.input_w

        flipped = False
