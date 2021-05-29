import yaml 
import os
import numpy as np
import pandas as pd
import glob
import tqdm

def main(dir):
    w=4096.0
    h=1536.0
    yamls = glob.glob(dir + '*.yaml')
    for fn in tqdm.tqdm(sorted(yamls)):
        with open(fn) as file:
            yaml_data=yaml.load(file,Loader=yaml.FullLoader)
        bboxes=[]
        class_id=[]
        for label in yaml_data:
            bboxes.append(label['bbox'])
            class_id.append(label['class'])

        
        np_box = np.array(bboxes, dtype=np.float64)
        np_box = xyxy2xywh(np_box)
        np_box[:,[0, 2]] /= w  # normalize x
        np_box[:,[1, 3]] /= h  # normalize y
        for index,id in enumerate(class_id):
            # box=' '.join(map(str,np_box[index]))
            # print(f'{class_id[index]} '+box)
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