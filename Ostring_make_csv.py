import pandas as pd
import os
def renamefiles():
    dir = "../DepthMapsOstringTest/map/"
    for root,dirs,files in os.walk(dir):
       for file in files:
           os.rename(root + file, root + str.zfill(file,14))
def main():
    dir = "../DepthMapsOstringTest/map/"
    for root,dirs,depth_files in os.walk(dir):
        img_files=depth_files
        csv = pd.DataFrame(list(zip(img_files,depth_files)),columns=['img','depthlabel'])
        csv.to_csv("OstringDataSet.csv",index=False)

if __name__=='__main__':
    main()