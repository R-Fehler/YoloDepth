from pathlib import Path
import pandas as pd
import os
import glob
from natsort import natsorted


def renamefiles():
    dir = "../DepthMapsOstringTest/map/"
    for root, dirs, files in os.walk(dir):
       for file in files:
           os.rename(root + file, root + str.zfill(file, 14))


def getListofFilesInDir(dir):
    return natsorted([file for file in os.listdir(dir) if os.path.isfile(os.path.join(dir, file))])


def getListofFileBaseNameInDir(dir):
    return natsorted([os.path.basename(file).split('.')[0] for file in os.listdir(dir) if os.path.isfile(os.path.join(dir, file))])


def checkAndFilterFilesWithSameNameExistInOtherDir(list, OtherDir, extension):
    return [file for file in list if os.path.exists(os.path.join(OtherDir, os.path.basename(file).split('.')[0]+extension))]


def main():
    img_dir = "./MapillaryVistasDataset/images/test"
    label_dir = "./MapillaryVistasDataset/labels/test"
    img_list = getListofFileBaseNameInDir(img_dir)
    label_list = getListofFileBaseNameInDir(label_dir)
    union_of_lists = list(set(img_list).intersection(label_list))
    union_of_lists = natsorted(union_of_lists)

    csv = pd.DataFrame(list(zip([i+'.jpg' for i in union_of_lists], [
                       i+'.txt' for i in union_of_lists])), columns=['img', 'bbox labels'])
    csv.to_csv("Mapillary_Vistas_Test.csv", index=False)


if __name__ == '__main__':
    main()
