import os
from pathlib import Path
from threading import Thread
import config
import torch
import sys
import glob
from torchsummary import summary

import numpy as np
import utils

from natsort import natsorted
from PIL import Image

from time import time
import matplotlib.pyplot as plt

from model import YOLOv3
from tqdm import tqdm
from utils import (
    load_checkpoint_transfer,
    mean_average_precision,
    get_evaluation_bboxes,
    save_checkpoint,
    load_checkpoint,
    check_class_accuracy,
    plot_couple_examples
)
from loss import YoloLoss
import warnings
warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = True



class InferenceHelper:
    def __init__(self, device='cuda:4', model_weights='yolov3_pascal_78.1map.pth.tar'):
        self.device = device
        model = YOLOv3(num_classes=config.NUM_CLASSES)
        checkpoint = torch.load(model_weights, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        summary(model, (3, config.IMAGE_SIZE, config.IMAGE_SIZE), device='cpu')
        model.eval()
        self.model = model.to(self.device)
        self.depth_target = torch.zeros((config.IMAGE_SIZE,config.IMAGE_SIZE))


    @torch.no_grad()
    def predict(self, inp):
    
        out = self.model(inp)
        return out


    def predictImgAndSave(self,filename,out_dir,conf_threshold, iou_threshold, anchors, depth_anchors):
        try:
            image = np.array(Image.open(filename).convert("RGB").resize((config.IMAGE_SIZE, config.IMAGE_SIZE)), dtype=np.dtype('float'))/255.0

        except FileNotFoundError:
            return 0

        image =np.transpose(image,(2,0,1))
        image = torch.tensor(image,device=self.device, dtype=torch.float)
        image = image.unsqueeze(0)
        predictions = self.predict(image)

        basename = os.path.basename(filename).split('.')[0]
        save_path = os.path.join(out_dir, basename + ".png")

        bboxes = []
        list_bboxes_at_scale=[]
        for i in range(3):
            S = predictions[i].shape[2]
            scaled_anchor_at_S = anchors[i]
            depth_anchors_at_S = depth_anchors[i]
            # list [100bbox,400bbox,1000bbox]

            boxes_scale_i = utils.cells_to_bboxes(
                predictions[i], scaled_anchor_at_S,depth_anchors=depth_anchors_at_S, S=S, is_preds=True
            ).tolist()
            list_bboxes_at_scale.append(boxes_scale_i)
            for idx, (box) in enumerate(boxes_scale_i):
                bboxes += box
        # we just want one bbox for each label, not one for each scale


        true_bboxes = []

        # take only bboxes prediction with valid depth prediction:
        bboxes_valid = [bbox for bbox in bboxes if bbox[6]>1] # depth must be greater than 1

        nms_all_boxes = utils.nms_torch(
            bboxes_valid,
            iou_threshold=iou_threshold,
            threshold=conf_threshold,
            # box_format=box_format,
        )

        list_bboxes_at_scale_idx=[list_bboxes_at_scale[i][0] for i in range(3)]
        list_nms_bboxes_at_scale=[utils.nms_torch(list_bboxes_at_scale_idx[i],iou_threshold,conf_threshold) for i in range(3)]
        # plot the depth predictions with bbox detection
        # TODO Idea plot conditional class distribution in the same fashion
        # plot(image,self.depth_target,list_bboxes_at_scale_idx,list_nms_bboxes_at_scale,nms_all_boxes,true_bboxes,save_path,basename)
        Thread(target=plot,args=(image,self.depth_target,list_bboxes_at_scale_idx,list_nms_bboxes_at_scale,nms_all_boxes,true_bboxes,save_path,basename)).start()
        return 1

        # Image.fromarray(final).save(save_path)
    @torch.no_grad()
    def predict_dir(self, test_dir, out_dir, conf_threshold, iou_threshold, anchors, depth_anchors):
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        all_files = natsorted(glob.glob(os.path.join(test_dir, "*")))
        self.model.eval()
        numfiles=0
        for fn in tqdm(all_files):
            numfiles+=self.predictImgAndSave(fn,out_dir, conf_threshold, iou_threshold, anchors, depth_anchors)
        print(f'{numfiles} images plotted')

    @torch.no_grad()        
    def predict_list_of_filepaths(self, fileList, out_dir, conf_threshold, iou_threshold, anchors, depth_anchors):
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        all_files = natsorted(fileList)
        self.model.eval()
        numfiles=0
        for fn in tqdm(all_files):
            numfiles+=self.predictImgAndSave(fn,out_dir, conf_threshold, iou_threshold, anchors, depth_anchors)
        print(f'{numfiles} images plotted')

            
def plot(img,depth_target,list_bboxes_at_scale_idx,list_nms_bboxes_at_scale,nms_all_boxes,true_bboxes,save_path,basename):
            fig=plt.figure(basename,figsize = (12,12),clear=True)
            plt.rc('font', size=2)
            plt.rc('axes',titlesize=14)
            plt.rc('xtick',labelsize =8)
            plt.rc('ytick',labelsize =8)
            utils.sparseDepthPredictionToDenseMap(depth_target, 0, config.TRAINING_EXAMPLES_PLOT_DIR_DEPTH,
                                            list_bboxes_at_scale_idx, list_nms_bboxes_at_scale,nms_all_boxes, true_bboxes,
                                            fig)

            # plot the bbox detection in rgb image
            img=img.squeeze()
            utils.plot_image(img.permute(1,2,0).detach().cpu(), nms_all_boxes,filename=save_path,figure=fig,target_boxes=true_bboxes,datasetName='')
            fig.suptitle(basename)
            fig.savefig(save_path,dpi=300)
            fig.clf()


def main(argv):
    start = time()
    inferHelper = InferenceHelper(config.DEVICE,model_weights='runs_mrtstorage/comparision_mAP_with_and_without_obj_depth_loss/from_PascalVOC_416px_ObjDepth_correct_depth_targets_transfer_toCompare_1/yolov3_pascal_78.1map_Dual_Dataset_Obj_specific_Depth_anchorDepths_batch_split_transfer_from_PASCALVOC_416px_only_obj_depth_loss_correct_targets.pth.tarMVDbestMAP_0.07487107068300247')

    scaled_anchors = (
        torch.tensor(config.ANCHORS)
        * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(config.DEVICE)

    depth_anchors = (
        torch.tensor(config.DEPTH_ANCHORS)
    ).to(config.DEVICE)

    if len(argv)>0:
        inferHelper.predict_list_of_filepaths(argv,
        './runs_mrtstorage/detections/from_cmd_line',
        config.INF_CONF_THRESHOLD,
        config.NMS_IOU_THRESH,
        scaled_anchors,
        depth_anchors,
        )
    # WARNING: execute exifautotran *   in your smartphone image folder beforehand.
    else:
        inferHelper.predict_dir(
        './TestInference_Thesis/', # in 
        "./runs_mrtstorage/detections/detections_thesis/", # out
        config.INF_CONF_THRESHOLD, 
        config.NMS_IOU_THRESH,
        scaled_anchors,
        depth_anchors
        )

    print(f"took :{time() - start}s")
    print(f"waiting for plotting threads to finish")

if __name__ == '__main__':
	main(sys.argv[1:])
    
