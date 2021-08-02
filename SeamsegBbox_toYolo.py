import yaml 
import os
import numpy as np
import pandas as pd
import glob
import tqdm

obj_threshold=0.78

aspectRatioPolesMin = 7

aggregate_like_PET = True
limit_aspect_ratio = True # skip poles that dont fit a bbox well, eg street light poles that are bend. PET doesnt work well with those because of clustering issues
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

yaml_labels_to_seamseg = {
    0:2,
    1:3,
    19:24,
    23:35,
    31:19,
    32:20,
    47:44,
    48:45,
    49:46,
    50:47,
    51:48,
    52:50,
    53:49,
    58:55
}

mvd_to_PET_aggregation = {
    20:20,
    21:20,
    22:20, # all poles of different kind
    23:23, # traffic lights
    24:25, # traffic sign front and back
    25:25,
}
# in seamseg yaml label files:
# // for some reason not the official seamseg class ids
# //enum class Seamseg : unsigned {
# // Curb = 0,
# // Fence = 1,
# // Marking = 19,
# // MarkingZebra = 35,
# // Person = 31,
# // Cyclist = 32,
# // StreetLight = 47,
# // Pole = 48,
# // TrafficSignFrame = 49,
# // UtilityPole = 50,
# // TrafficLight = 51,
# // TrafficSignFront = 52,
# // TrafficSignBack = 53,
# // Car = 58
# //};



# enum class Seamseg : unsigned {
# Curb = 2,
# Fence = 3,
# Marking = 24,
# MarkingZebra = 23,
# Person = 19,
# Cyclist = 20,
# StreetLight = 44,
# Pole = 45,
# TrafficSignFrame = 46,
# UtilityPole = 47,
# TrafficLight = 48,
# TrafficSignBack = 49,
# TrafficSignFront = 50,
# Car = 55
# };


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

            # TODO idea: label also the objectness, modeling the pseudo label as uncertain measurement
            if objectness[index] > obj_threshold:
                if aggregate_like_PET:
                    try:
                        class_id[index] = mvd_to_PET_aggregation[seamseg_to_mvd_classlabels[yaml_labels_to_seamseg[class_id[index]]]]
                    except KeyError:
                        try:
                            class_id[index] = mvd_to_PET_aggregation[seamseg_to_mvd_classlabels[class_id[index]]]
                        except KeyError:
                            continue
                        
                else:
                    try:
                        class_id[index] = seamseg_to_mvd_classlabels[yaml_labels_to_seamseg[class_id[index]]]
                    except KeyError:
                        try:
                            class_id[index] = seamseg_to_mvd_classlabels[class_id[index]]
                        except KeyError:
                            continue
                
                if limit_aspect_ratio:
                    if class_id[index] == 20:
                        aspectRatio=np_box[index,3]/np_box[index,2]
                        if aspectRatio < aspectRatioPolesMin:
                            # print(f'skipping pole cls id {class_id[index]} with aspect ratio: {aspectRatio}')
                            continue
                        
                    # print(class_id[index])
                    with open( fn + '.txt', 'a') as file:
                        file.write('%g %.6f %.6f %.6f %.6f\n' % (class_id[index], *np_box[index]))



                else:
                    with open( fn + '.txt', 'a') as file:
                        file.write('%g %.6f %.6f %.6f %.6f\n' % (class_id[index], *np_box[index]))

            
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