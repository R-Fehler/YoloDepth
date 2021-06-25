import yaml 
import os
import numpy as np
import pandas as pd
import glob
import tqdm

obj_threshold=0.8


seamseg_to_mvd_classlabels = {
    0: 0,
    1: 1,
    8: 2,
    19: 3,
    20: 4,
    21: 5,
    22: 6,
    23: 7,
    32: 8,
    33: 9,
    34: 10,
    35: 11,
    36: 12,
    37: 13,
    38: 14,
    39: 15,
    40: 16,
    41: 17,
    42: 18,
    44: 19,
    45: 20,
    46: 21,
    47: 22,
    48: 23,
    49: 24,
    50: 25,
    51: 26,
    52: 27,
    53: 28,
    54: 29,
    55: 30,
    56: 31,
    57: 32,
    58: 33,
    59: 33,
    60: 34,
    61: 35,
    62: 36
}

def main(dir):
    w=4096.0
    h=1536.0
    yamls = glob.glob(dir + '*.yaml')
    for fn in tqdm.tqdm(sorted(yamls)):
        with open(fn) as file:
            yaml_data=yaml.load(file,Loader=yaml.FullLoader)
        bboxes=[]
        class_id=[]
        objectness=[]
        for detection in yaml_data:
            bboxes.append(detection['bbox'])
            class_id.append(detection['class'])
            objectness.append(detection['objectness'])

        
        np_box = np.array(bboxes, dtype=np.float64)
        np_box = xyxy2xywh(np_box)
        np_box[:,[0, 2]] /= w  # normalize x
        np_box[:,[1, 3]] /= h  # normalize y
        for index,id in enumerate(class_id):
            # box=' '.join(map(str,np_box[index]))
            # print(f'{class_id[index]} '+box)
            if objectness[index]>obj_threshold:
                try:
                    with open( fn + '.txt', 'a') as file:
                        file.write('%g %.6f %.6f %.6f %.6f\n' % (seamseg_to_mvd_classlabels[class_id[index]], *np_box[index]))
                except KeyError:
                    pass
            
def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y
if __name__=='__main__':
    main('/home/fehler/yoloDepth/seamseg_boxes/')