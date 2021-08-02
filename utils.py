import shutil
from threading import Thread
from PIL import Image
from matplotlib import colors
import matplotlib
from matplotlib.colors import ListedColormap
from numpy.lib.function_base import append
from torch._C import dtype
from torch.serialization import load
from tqdm import trange
import config
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import random
import torch
import torchvision.ops
import math 

import soft_nms

from pathlib import Path
from scipy.interpolate import griddata
import turbo_colormap
from collections import Counter
from torch.utils.data import DataLoader, dataloader
from tqdm import tqdm


def iou_width_height(boxes1, boxes2):
	"""
	Parameters:
		boxes1 (tensor): width and height of the first bounding boxes
		boxes2 (tensor): width and height of the second bounding boxes
	Returns:
		tensor: Intersection over union of the corresponding boxes
	"""
	intersection = torch.min(boxes1[..., 0], boxes2[..., 0]) * torch.min(
		boxes1[..., 1], boxes2[..., 1]
	)
	union = (
		boxes1[..., 0] * boxes1[..., 1] + boxes2[..., 0] * boxes2[..., 1] - intersection
	)
	return intersection / union


def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
	"""
	Video explanation of this function:
	https://youtu.be/XXYG5ZWtjj0

	This function calculates intersection over union (iou) given pred boxes
	and target boxes.

	Parameters:
		boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
		boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
		box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)

	Returns:
		tensor: Intersection over union for all examples
	"""

	if box_format == "midpoint":
		box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
		box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
		box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
		box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
		box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
		box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
		box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
		box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

	if box_format == "corners":
		box1_x1 = boxes_preds[..., 0:1]
		box1_y1 = boxes_preds[..., 1:2]
		box1_x2 = boxes_preds[..., 2:3]
		box1_y2 = boxes_preds[..., 3:4]
		box2_x1 = boxes_labels[..., 0:1]
		box2_y1 = boxes_labels[..., 1:2]
		box2_x2 = boxes_labels[..., 2:3]
		box2_y2 = boxes_labels[..., 3:4]

	x1 = torch.max(box1_x1, box2_x1)
	y1 = torch.max(box1_y1, box2_y1)
	x2 = torch.min(box1_x2, box2_x2)
	y2 = torch.min(box1_y2, box2_y2)

	intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
	box1_area = torch.abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
	box2_area = torch.abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

	return intersection / (box1_area + box2_area - intersection + 1e-6)

def nms_torch(bboxes, iou_threshold, threshold):
# assert type(bboxes) == list
	bboxes = sorted(bboxes, key=lambda x: x[1],reverse=True)
	bboxes_limit=config.MAX_NO_OF_DETECTIONS
	bboxes = bboxes[:bboxes_limit]
	bboxes = [box for box in bboxes if box[1] > threshold]

	if(len(bboxes)<1):
		return bboxes
	if config.USE_SOFT_NMS:
		bboxes_tensor=torch.tensor(bboxes,device=config.DEVICE)
		soft_indxs=[]
		# for cls in range(config.NUM_CLASSES):
		# bboxes_cls_tensor = bboxes_tensor[bboxes_tensor[0]==cls]
		bbox_coords = bboxes_tensor[...,2:6]
		bbox_coords = torchvision.ops.box_convert(bbox_coords,'cxcywh','xyxy')
		bbox_scores = bboxes_tensor[...,1]
		soft_indxs=soft_nms.soft_nms_pytorch(bbox_coords,bbox_scores,thresh=threshold,cuda=True)
		bboxes_after_nms=[ bboxes[indx] for indx in soft_indxs.tolist() ]
		return bboxes_after_nms
		
	bboxes_after_nms = []
	bbox_coords=[box[2:6] for box in bboxes]
	bbox_scores=[box[1] for box in bboxes]
	bbox_classes=[box[0]for box in bboxes]
	bbox_coords = torch.tensor(bbox_coords).to(config.DEVICE)
	bbox_scores = torch.tensor(bbox_scores).to(config.DEVICE)
	bbox_classes = torch.tensor(bbox_classes).to(config.DEVICE)
	if(len(bboxes)<1):
		return bboxes
	else:
		bbox_coords = torchvision.ops.box_convert(bbox_coords,'cxcywh','xyxy')
		# batched nms for comparing only bboxes of the same class category /id
		indxs=torchvision.ops.batched_nms(bbox_coords,bbox_scores,bbox_classes,iou_threshold=iou_threshold)
		bboxes_after_nms =[bboxes[indx] for indx in indxs]


	# while bboxes:
	#     chosen_box = bboxes.pop(0)

	#     bboxes = [
	#         box
	#         for box in bboxes
	#         if box[0] != chosen_box[0]
	#         or intersection_over_union(
	#             torch.tensor(chosen_box[2:]),
	#             torch.tensor(box[2:]),
	#             box_format=box_format,
	#         )
	#         < iou_threshold
	#     ]

	#     bboxes_after_nms.append(chosen_box)
	#     # print(len(bboxes_after_nms))


		return bboxes_after_nms

def non_max_suppression(bboxes, iou_threshold, threshold, box_format="corners"):
	"""
	Video explanation of this function:
	https://youtu.be/YDkjWEN8jNA

	Does Non Max Suppression given bboxes

	Parameters:
		bboxes (list): list of lists containing all bboxes with each bboxes
		specified as [class_pred, prob_score, x1, y1, x2, y2]
		iou_threshold (float): threshold where predicted bboxes is correct
		threshold (float): threshold to remove predicted bboxes (independent of IoU)
		box_format (str): "midpoint" or "corners" used to specify bboxes

	Returns:
		list: bboxes after performing NMS given a specific IoU threshold
	"""

	assert type(bboxes) == list

	bboxes = [box for box in bboxes if box[1] > threshold]
	bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
	bboxes_after_nms = []

	while bboxes:
		chosen_box = bboxes.pop(0)

		bboxes = [
			box
			for box in bboxes
			if box[0] != chosen_box[0]
			or intersection_over_union(
				torch.tensor(chosen_box[2:]),
				torch.tensor(box[2:]),
				box_format=box_format,
			)
			< iou_threshold
		]

		bboxes_after_nms.append(chosen_box)

	return bboxes_after_nms

# #TODO
# def evaluateDepth(pred_boxes, true_boxes, iou_threshold = 0.5, num_classes = 37):
# 	average_precisions = []
# 	'''
# 	evaluates the SILog Loss only on pred_boxes depths which are the non max supressed boxes from 3 scale predictions.
# 	return var, mean and SILog Loss for objects. as cls wise vector entries
# 	'''
	
# 	# used for numerical stability later on
# 	epsilon = 1e-6
# 	var = 0
# 	mean = 0
# 	for cls in range(num_classes):
# 		detections = []
# 		ground_truths = []

# 		# Go through all predictions and targets,
# 		# and only add the ones that belong to the
# 		# current class c
# 		for detection in pred_boxes:
# 			if detection[1] == cls:
# 				detections.append(detection)

# 		for true_box in true_boxes:
# 			if true_box[1] == cls:
# 				ground_truths.append(true_box)

# 		# find the amount of bboxes for each training example
# 		# Counter here finds how many ground truth bboxes we get
# 		# for each training example, so let's say img 0 has 3,
# 		# img 1 has 5 then we will obtain a dictionary with:
# 		# amount_bboxes = {0:3, 1:5}

# 		# count how many instances of each class we have in each image. images indexes are keys and no of gt boxes are vals
# 		amount_bboxes_target = Counter([gt[0] for gt in ground_truths])

# 		# We then go through each key, val in this dictionary
# 		# and convert to the following (w.r.t same example):
# 		# ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
# 		for key, val in amount_bboxes_target.items():
# 			amount_bboxes_target[key] = torch.zeros(val)

# 		# sort by box probabilities which is index 2
# 		detections.sort(key=lambda x: x[2], reverse=True)
# 		TP = torch.zeros((len(detections)))
# 		FP = torch.zeros((len(detections)))
# 		total_true_bboxes = len(ground_truths)

# 		# If none exists for this class then we can safely skip
# 		if total_true_bboxes == 0:
# 			continue

# 		for detection_idx, detection in enumerate(detections):
# 			# Only take out the ground_truths that have the same
# 			# training idx as detection
# 			ground_truth_img = [
# 				bbox for bbox in ground_truths if bbox[0] == detection[0]
# 			]

# 			num_gts = len(ground_truth_img)
# 			best_iou = 0

# 			for idx, gt in enumerate(ground_truth_img):
# 				iou = intersection_over_union(
# 					torch.tensor(detection[3:]),
# 					torch.tensor(gt[3:]),
# 					box_format=box_format,
# 				)

# 				if iou > best_iou:
# 					best_iou = iou
# 					best_gt_idx = idx

# 			if best_iou > iou_threshold:
# 				# only detect ground truth detection once
# 				if amount_bboxes_target[detection[0]][best_gt_idx] == 0:
# 					# true positive and add this bounding box to seen
# 					TP[detection_idx] = 1
# 					depth_difference = detection[-1] - gt[-1]
# 					csv_line=f'{cls}, {depth_difference}'
# 					print(csv_line)
# 					with open() as f:
# 						f.write(csv_line)
# 					amount_bboxes_target[detection[0]][best_gt_idx] = 1
# 				else:
# 					FP[detection_idx] = 1

# 			# if IOU is lower then the detection is a false positive
# 			else:
# 				FP[detection_idx] = 1
# 			#TODO class wise Depth TP/FP mean and variance of error
# 		TP_cumsum = torch.cumsum(TP, dim=0)
# 		FP_cumsum = torch.cumsum(FP, dim=0)
# 		recalls = TP_cumsum / (total_true_bboxes + epsilon)
# 		precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
# 		precisions = torch.cat((torch.tensor([1]), precisions))
# 		recalls = torch.cat((torch.tensor([0]), recalls))
# 		# torch.trapz for numerical integration
# 		average_precisions.append(torch.trapz(precisions, recalls))
# 	return var, mean


# def evaluateYOLOV5():
# 	# Dataloader
#     if not training:
#         if device.type != 'cpu':
#             model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
#         task = opt.task if opt.task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
#         dataloader = create_dataloader(data[task], imgsz, batch_size, gs, opt, pad=0.5, rect=True,
#                                        prefix=colorstr(f'{task}: '))[0]

#     seen = 0
#     names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
#     s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
#     p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
#     loss = torch.zeros(3, device=device)
#     jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []
#     for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
#         img = img.to(device, non_blocking=True)
#         img = img.half() if half else img.float()  # uint8 to fp16/32
#         img /= 255.0  # 0 - 255 to 0.0 - 1.0
#         targets = targets.to(device)
#         nb, _, height, width = img.shape  # batch size, channels, height, width

#         with torch.no_grad():
#             # Run model
#             t = time_synchronized()
#             out, train_out = model(img, augment=augment)  # inference and training outputs
#             t0 += time_synchronized() - t

#             # Compute loss
#             if compute_loss:
#                 loss += compute_loss([x.float() for x in train_out], targets)[1][:3]  # box, obj, cls

#             # Run NMS
#             targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
#             lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
#             t = time_synchronized()
#             out = non_max_suppression(out, conf_thres=conf_thres, iou_thres=iou_thres, labels=lb, multi_label=True)
#             t1 += time_synchronized() - t

#         # Statistics per image
#         for si, pred in enumerate(out):
#             labels = targets[targets[:, 0] == si, 1:]
#             nl = len(labels)
#             tcls = labels[:, 0].tolist() if nl else []  # target class
#             path = Path(paths[si])
#             seen += 1

#             if len(pred) == 0:
#                 if nl:
#                     stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
#                 continue

#             # Predictions
#             predn = pred.clone()
#             scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred

#             # Append to text file
#             if save_txt:
#                 gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]  # normalization gain whwh
#                 for *xyxy, conf, cls in predn.tolist():
#                     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
#                     line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
#                     with open(save_dir / 'labels' / (path.stem + '.txt'), 'a') as f:
#                         f.write(('%g ' * len(line)).rstrip() % line + '\n')

#             # W&B logging
#             if plots and len(wandb_images) < log_imgs:
#                 box_data = [{"position": {"minX": xyxy[0], "minY": xyxy[1], "maxX": xyxy[2], "maxY": xyxy[3]},
#                              "class_id": int(cls),
#                              "box_caption": "%s %.3f" % (names[cls], conf),
#                              "scores": {"class_score": conf},
#                              "domain": "pixel"} for *xyxy, conf, cls in pred.tolist()]
#                 boxes = {"predictions": {"box_data": box_data, "class_labels": names}}  # inference-space
#                 wandb_images.append(wandb.Image(img[si], boxes=boxes, caption=path.name))

#             # Append to pycocotools JSON dictionary
#             if save_json:
#                 # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
#                 image_id = int(path.stem) if path.stem.isnumeric() else path.stem
#                 box = xyxy2xywh(predn[:, :4])  # xywh
#                 box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
#                 for p, b in zip(pred.tolist(), box.tolist()):
#                     jdict.append({'image_id': image_id,
#                                   'category_id': coco91class[int(p[5])] if is_coco else int(p[5]),
#                                   'bbox': [round(x, 3) for x in b],
#                                   'score': round(p[4], 5)})

#             # Assign all predictions as incorrect
#             correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
#             if nl:
#                 detected = []  # target indices
#                 tcls_tensor = labels[:, 0]

#                 # target boxes
#                 tbox = xywh2xyxy(labels[:, 1:5])
#                 scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels
#                 if plots:
#                     confusion_matrix.process_batch(predn, torch.cat((labels[:, 0:1], tbox), 1))

#                 # Per target class
#                 for cls in torch.unique(tcls_tensor):
#                     ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
#                     pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices

#                     # Search for detections
#                     if pi.shape[0]:
#                         # Prediction to target ious
#                         ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices

#                         # Append detections
#                         detected_set = set()
#                         for j in (ious > iouv[0]).nonzero(as_tuple=False):
#                             d = ti[i[j]]  # detected target
#                             if d.item() not in detected_set:
#                                 detected_set.add(d.item())
#                                 detected.append(d)
#                                 correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
#                                 if len(detected) == nl:  # all targets already located in image
#                                     break

#             # Append statistics (correct, conf, pcls, tcls)
#             stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))


# super slow when there are many detections of a cls, eg cls 20 takes forever
def mean_average_precision(
	pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=37, reduction=True, filename_dict={}, 
):
	"""
	Video explanation of this function:
	https://youtu.be/FppOzcDvaDI

	This function calculates mean average precision (mAP)
	train_idx is the idx corresponding to the image that the box is detected in
	Parameters:
		pred_boxes (list): list of lists containing all bboxes with each bboxes
		specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
		true_boxes (list): Similar as pred_boxes except all the correct ones
		iou_threshold (float): threshold where predicted bboxes is correct
		box_format (str): "midpoint" or "corners" used to specify bboxes
		num_classes (int): number of classes

	Returns:
		float: mAP value across all classes given a specific IoU threshold
	"""
	# converted_bboxes = torch.cat((best_class, scores, x, y, w_h,z), dim=-1).reshape(BATCH_SIZE, num_anchors * S * S, 7)

	# list storing all AP for respective classes
	average_precisions = []

	# used for numerical stability later on
	epsilon = 1e-6

	for class_id in trange(num_classes,desc='cls wise evaluation of mAP'):
		detections = []
		ground_truths = []

		# Go through all predictions and targets,
		# and only add the ones that belong to the
		# current class c
		for detection in pred_boxes:
			if detection[1] == class_id:
				detections.append(detection)

		for true_box in true_boxes:
			if true_box[1] == class_id:
				ground_truths.append(true_box)

		# find the amount of bboxes for each training example
		# Counter here finds how many ground truth bboxes we get
		# for each training example, so let's say img 0 has 3,
		# img 1 has 5 then we will obtain a dictionary with:
		# amount_bboxes = {0:3, 1:5}
		amount_bboxes = Counter([gt[0] for gt in ground_truths])

		# We then go through each key, val in this dictionary
		# and convert to the following (w.r.t same example):
		# ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
		for key, val in amount_bboxes.items():
			amount_bboxes[key] = torch.zeros(val)

		# sort by box probabilities which is index 2
		detections.sort(key=lambda x: x[2], reverse=True)
		TruePositives = torch.zeros((len(detections)))
		FalsePositives = torch.zeros((len(detections)))
		total_true_bboxes = len(ground_truths)

		# If none exists for this class then we can safely skip
		if total_true_bboxes == 0:
			average_precisions.append(0) # to keep list index and class index in line
			continue

		for detection_idx, detection in enumerate(tqdm(detections,leave=False,desc='detections of cls')):
			# Only take out the ground_truths that have the same
			# training idx as detection
			ground_truth_boxes_in_img = [
				bbox for bbox in ground_truths if bbox[0] == detection[0]
			]

			num_gts = len(ground_truth_boxes_in_img)
			best_iou = 0
			# det_box_conv=torch.tensor(detection[3:7],device=config.DEVICE).unsqueeze_(0)
			# det_box_conv=torchvision.ops.box_convert(det_box_conv,'cxcywh','xyxy') # convert to xyxy format
			# gt_boxes_conv=torch.tensor([box[3:7] for box in ground_truth_boxes_in_img],device=config.DEVICE)
			# gt_boxes_conv=torchvision.ops.box_convert(gt_boxes_conv,'cxcywh','xyxy')
			# IOU_matrix=torchvision.ops.box_iou(det_box_conv,gt_boxes_conv)
			# best_gt_idx=torch.argmax(IOU_matrix).item()
			# best_iou=torch.max(IOU_matrix).item()

			#TODO use vectorized box_iou capability no for loop necessary
			for idx, gt_box in enumerate(ground_truth_boxes_in_img):
				iou = intersection_over_union(
					torch.tensor(detection[3:]),
					torch.tensor(gt_box[3:]),
					box_format=box_format,
				)
				# gt_box_conv=torch.tensor(gt_box[3:7]).unsqueeze_(0)
				# gt_box_conv=torchvision.ops.box_convert(gt_box_conv,'cxcywh','xyxy')
				# iou = torchvision.ops.box_iou(det_box_conv,gt_box_conv)
				if iou > best_iou:
					best_iou = iou
					best_gt_idx = idx

			if best_iou > iou_threshold:
				gt_box=ground_truth_boxes_in_img[best_gt_idx]
				# only detect ground truth detection once
				if amount_bboxes[detection[0]][best_gt_idx] == 0:
					# true positive and add this bounding box to seen
					TruePositives[detection_idx] = 1
					amount_bboxes[detection[0]][best_gt_idx] = 1

					## TODO if difference is greater than 100% plot GT in Green in Image with ID and depth and plot detection in red with ID with id and depth
					if(config.EVAL_DEPTH_ERROR):
						if(gt_box[-1]>config.MIN_DEPTH_EVAL and gt_box[-1]<config.MAX_DEPTH_EVAL): # only for valid Depth GT evaluate estimate
							#     depth_difference = obj_pred_depths - obj_depth_target in the loss fkt

							depth_difference = detection[-1] - gt_box[-1] # depth is last value
							relative_depth_difference = abs(detection[-1] - gt_box[-1])/gt_box[-1] # depth is last value
							depth_difference_log = math.log(detection[-1]) - math.log(gt_box[-1])
							relative_depth_difference_log = abs(math.log(detection[-1]) - math.log(gt_box[-1]))/math.log(gt_box[-1])
							
							if(relative_depth_difference>config.PLOT_EVAL_ERROR_THRESHOLD and config.PLOT_EVAL_ERROR):
								fig=plt.figure(figsize = (6,6))
								plt.rc('font',size=2) 
								plt.rc('axes',titlesize=6)
								filename=filename_dict[gt_box[0]]
								img_path = os.path.join(config.OSTRING_DS.IMAGE_DIR,filename)

								image = Image.open(img_path).convert("RGB").resize((config.IMAGE_SIZE,config.IMAGE_SIZE))
								height = config.IMAGE_SIZE
								width = config.IMAGE_SIZE
								cmap1 = plt.get_cmap("tab20")
								cmap2 = plt.get_cmap("tab20c")			
								colors1 = [cmap1(i) for i in np.linspace(0, 1, 20)]
								colors2 = [cmap2(i) for i in np.linspace(0,1,config.NUM_CLASSES-20)]
								colors1.extend(colors2)
								clscolors=colors1
								ax = fig.add_subplot(1,1,1)
								ax.imshow(image)
								className=config.MVD_CLASSES[class_id]
								ax.set_title(f'IOU:{best_iou.item():.2f}% Class:{className} Conf:{detection[2]:.2f}% \nPrediction in green:{detection[-1]:.1f}m P-Label in red:{gt_box[-1]:.1f}m\n Relative Error:{relative_depth_difference*100:.1f}% ')
								def plot(plotBox,ax,color='r'):
									class_pred = plotBox[1]
									conf = plotBox[2]
									box = plotBox[3:]
									x_img = box[0]
									y_img = box[1]
									box_width = box[2]
									box_height = box[3]
									z = box[4]

									upper_left_x = box[0] - box[2] / 2
									upper_left_y = box[1] - box[3] / 2
									rect = patches.Rectangle(
									(upper_left_x * width, upper_left_y * height),
									box[2] * width,
									box[3] * height,
									linewidth=1, # plot line thickness as confidence
									edgecolor=color,
									facecolor="none",
									)
									# Add the patch to the Axes
									ax.add_patch(rect)
									# ax.text(
									# upper_left_x * width,
									# upper_left_y * height,
									# # s="c:" + str(int(class_pred)) + f"d:{box[4]:.1f} cf:{conf:.2f}" ,
									# s=f"d:{box[4]:.1f}",
									# color="white",
									# verticalalignment="top",
									# bbox={"color": color, "pad": 0},
									# )
								plot(gt_box,ax,color='r')
								plot(detection,ax,color='g')
								plt.savefig(config.EVAL_PLOTS_DIR+filename+f'_{class_id}_{detection_idx}.png',dpi=300)
								plt.clf()

							bbox_GT = gt_box[3:6]
							bbox_Pred = detection[3:6]
							boxStringGT = ','.join(str(x) for x in bbox_GT)
							boxStringPred = ','.join(str(x) for x in bbox_Pred)
							csv_line = f'{class_id}, {gt_box[-1]}, {detection[-1]}, {relative_depth_difference}\n'
							csv_line_box = f'{class_id}, {gt_box[-1]}, {detection[-1]}, {relative_depth_difference},{boxStringGT},{boxStringPred}\n'
							
							csv_line_log=f'{class_id}, {gt_box[-1]}, {detection[-1]}, {relative_depth_difference_log}\n'
							csv_line_log_box=f'{class_id}, {gt_box[-1]}, {detection[-1]}, {relative_depth_difference_log},{boxStringGT},{boxStringPred}\n'
							with open(config.EVAL_FILE_NAME,mode='a') as f:
								f.write(csv_line)
							with open('log_'+config.EVAL_FILE_NAME,mode='a') as f:
								f.write(csv_line_log)



				else:
					FalsePositives[detection_idx] = 1

			# if IOU is lower then the detection is a false positive
			else:
				FalsePositives[detection_idx] = 1
			#TODO class wise Depth TP/FP mean and variance of error
		TP_cumsum = torch.cumsum(TruePositives, dim=0)
		FP_cumsum = torch.cumsum(FalsePositives, dim=0)
		recalls = TP_cumsum / (total_true_bboxes + epsilon)
		precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
		precisions = torch.cat((torch.tensor([1]), precisions))
		recalls = torch.cat((torch.tensor([0]), recalls))
		# torch.trapz for numerical integration
		average_precisions.append(torch.trapz(precisions, recalls))
	else:
		return sum(average_precisions) / len(average_precisions), average_precisions

def plot_image(image, pred_boxes,filename='default.jpg',figure=None,target_boxes=None,datasetName=''):
	"""Plots predicted bounding boxes on the image"""
	cmap1 = plt.get_cmap("tab20")
	cmap2 = plt.get_cmap("tab20c")
	# if config.DATASET=='COCO':
	#     class_labels = config.COCO_LABELS
	# elif config.DATASET=='PASCAL':
	#     class_labels = config.PASCAL_CLASSES
	# elif config.DATASET == 'Mapillary':
	#     class_labels = config.MVD_CLASSES
	# else:
	class_labels = config.MVD_CLASSES

# combine them and build a new colormap
	colors1 = [cmap1(i) for i in np.linspace(0, 1, 20)]
	colors2 = [cmap2(i) for i in np.linspace(0,1,config.NUM_CLASSES-20)]
	colors1.extend(colors2)
	clscolors=colors1
	im = np.array(image)
	height, width, _ = im.shape



	# Create figure and axes
	if figure != None:
		fig = figure
	else:
		fig = plt.figure()
	ax1 = fig.add_subplot(3, 3,1)
	imgplot = ax1.imshow(im)

	ax3 = fig.add_subplot(3,3,7)
	ax4 = fig.add_subplot(3,3,8)
	ax3.set_title('BEV Prediction')
	ax4.set_title('BEV Target')
	def convertPixelCoordsToBEV(x_img,z,dataset=datasetName):
			#fisheye model
		if('Ostring' in datasetName):
			f = 1 # model says 9000
			x = z * math.tan(x_img-0.5)
		else:
			# pinhole
			f = 1
			x = (z/f) * (x_img-0.5)
			y = (z/f) * (y_img-0.5)
		
		
		return x
		
	# box[0] is x midpoint, box[2] is width
	# box[1] is y midpoint, box[3] is height

	# Create a Rectangle patch
	max_x = 0
	noOfInvalid = 0


	if not config.PLOT_CAR_BOXES_NON_LOG_SPACE:
		for box in pred_boxes:
			assert len(box) == 7, "box should contain class pred, confidence, x, y, width, height and depth"
			class_pred = box[0]
			conf = box[1]
			box = box[2:]
			x_img = box[0]
			y_img = box[1]
			box_width = box[2]
			box_height = box[3]
			z = box[4]
			if z < 1:
				noOfInvalid += 1
			x = convertPixelCoordsToBEV(x_img,z)
			x1 = convertPixelCoordsToBEV(x_img-box_width*config.OBJ_WIDTH_BEV_SCALING,z)
			x2 = convertPixelCoordsToBEV(x_img+box_width*config.OBJ_WIDTH_BEV_SCALING,z)
			max_x = abs(x) if abs(x) > max_x else max_x
			ax3.scatter(x,z,color=clscolors[int(class_pred)],s=config.MARKER_SCALING*conf,marker='.')
			ax3.plot([x1,x2],[z,z],color=clscolors[int(class_pred)],linewidth=conf)
			upper_left_x = box[0] - box[2] / 2
			upper_left_y = box[1] - box[3] / 2
			rect = patches.Rectangle(
				(upper_left_x * width, upper_left_y * height),
				box[2] * width,
				box[3] * height,
				linewidth=0.7*box[1], # plot line thickness as confidence
				edgecolor=clscolors[int(class_pred)],
				facecolor="none",
			)
			# Add the patch to the Axes
			ax1.add_patch(rect)
			ax1.text(
				upper_left_x * width,
				upper_left_y * height,
				s="c:" + str(int(class_pred)) + f"d:{box[4]:.1f} cf:{conf:.2f}" ,
				# s=f"d:{box[4]:.1f}",
				color="white",
				verticalalignment="top",
				bbox={"color": clscolors[int(class_pred)], "pad": 0},
			)
		ax1.set_title(f'Prediction')
		ax3.set_yscale('log')
		ax3.set_ylim([1,255])
		ax3.set_yticks([2,5,10,20,30,40,50,100,200])
		ax3.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

		ax2 = fig.add_subplot(3, 3,2)
		imgplot = ax2.imshow(im)

		noOfInvalid = 0
		target_boxes =[bbox for bbox in target_boxes if bbox[1]!=0]

		for box in target_boxes:
			assert len(box) == 7, "box should contain class pred, confidence, x, y, width, height and depth"
			class_pred = box[0]
			box = box[2:]
			x_img = box[0]
			y_img = box[1]
			box_width = box[2]
			box_height = box[3]
			z = box[4]
			if z < 1:
				noOfInvalid += 1
			x= convertPixelCoordsToBEV(x_img,z)
			x1= convertPixelCoordsToBEV(x_img-box_width*config.OBJ_WIDTH_BEV_SCALING,z)
			x2= convertPixelCoordsToBEV(x_img+box_width*config.OBJ_WIDTH_BEV_SCALING,z)
			max_x = abs(x) if abs(x) > max_x else max_x

			ax4.scatter(x,z,color=clscolors[int(class_pred)],marker='.',s=config.MARKER_SCALING*1)
			ax4.plot([x1,x2],[z,z],color=clscolors[int(class_pred)],linewidth=0.5)

			upper_left_x = box[0] - box[2] / 2
			upper_left_y = box[1] - box[3] / 2
			rect = patches.Rectangle(
				(upper_left_x * width, upper_left_y * height),
				box[2] * width,
				box[3] * height,
				linewidth=0.5,
				edgecolor=clscolors[int(class_pred)],
				facecolor="none",
			)
			# Add the patch to the Axes
			ax2.add_patch(rect)
			ax2.text(
				upper_left_x * width,
				upper_left_y * height,
				s="c:" + str(int(class_pred)) + f"d:{box[4]:.1f}" ,
				# s=f"d:{box[4]:.1f}",
				color="white",
				verticalalignment="top",
				bbox={"color": clscolors[int(class_pred)], "pad": 0},

			)
		ax2.set_title(f'Target, ratio of  invalid/valid depth : {noOfInvalid}/{len(target_boxes)}')
		if config.FIX_X_LIMIT == True:
			ax3.set_xlim([- config.FIX_BEV_X_LIM,config.FIX_BEV_X_LIM])
			ax4.set_xlim([- config.FIX_BEV_X_LIM,config.FIX_BEV_X_LIM])
		else:
			ax3.set_xlim([-max_x-1,+max_x+1])
			ax4.set_xlim([-max_x-1,+max_x+1])
		ax4.set_yscale('log')
		ax4.set_ylim([1,255])
		ax4.set_yticks([2,5,10,20,30,40,50,100,200])
		ax4.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())



	else:
		for box in pred_boxes:
			assert len(box) == 7, "box should contain class pred, confidence, x, y, width, height and depth"
			class_pred = box[0]
			conf = box[1]
			box = box[2:]
			x_img = box[0]
			y_img = box[1]
			box_width = box[2]
			box_height = box[3]
			z = box[4]
			if z < 1:
				noOfInvalid += 1
			x = convertPixelCoordsToBEV(x_img,z)
			x1 = convertPixelCoordsToBEV(x_img-box_width*config.OBJ_WIDTH_BEV_SCALING,z)
			x2 = convertPixelCoordsToBEV(x_img+box_width*config.OBJ_WIDTH_BEV_SCALING,z)
			max_x = abs(x) if abs(x) > max_x else max_x
			ax3.scatter(x,z,color=clscolors[int(class_pred)],s=config.MARKER_SCALING*conf,marker='.')
			ax3.plot([x1,x2],[z,z],color=clscolors[int(class_pred)],linewidth=conf)
			if class_pred == 30:
				carRect = patches.Rectangle(
					(x1,z), # car is ~ 4m long
					x2-x1,
					4,
					linewidth=conf, # plot line thickness as confidence
					edgecolor=clscolors[int(class_pred)],
					facecolor='none',

				)
				ax3.add_patch(carRect)
			
				
			upper_left_x = box[0] - box[2] / 2
			upper_left_y = box[1] - box[3] / 2
			rect = patches.Rectangle(
			(upper_left_x * width, upper_left_y * height),
			box[2] * width,
			box[3] * height,
			linewidth=0.7*box[1], # plot line thickness as confidence
			edgecolor=clscolors[int(class_pred)],
			facecolor="none",
			)
			# Add the patch to the Axes
			ax1.add_patch(rect)
			ax1.text(
			upper_left_x * width,
			upper_left_y * height,
			s="c:" + str(int(class_pred)) + f"d:{box[4]:.1f} cf:{conf:.2f}" ,
			# s=f"d:{box[4]:.1f}",
			color="white",
			verticalalignment="top",
			bbox={"color": clscolors[int(class_pred)], "pad": 0},
			)
		ax1.set_title(f'Prediction')
		ax3.set_ylim([1,100])

		ax2 = fig.add_subplot(3, 3,2)
		imgplot = ax2.imshow(im)

		noOfInvalid = 0
		target_boxes =[bbox for bbox in target_boxes if bbox[1]!=0]

		for box in target_boxes:
			assert len(box) == 7, "box should contain class pred, confidence, x, y, width, height and depth"
			class_pred = box[0]
			box = box[2:]
			x_img = box[0]
			y_img = box[1]
			box_width = box[2]
			box_height = box[3]
			z = box[4]
			if z < 1:
				noOfInvalid += 1
			x= convertPixelCoordsToBEV(x_img,z)
			x1= convertPixelCoordsToBEV(x_img-box_width*config.OBJ_WIDTH_BEV_SCALING,z)
			x2= convertPixelCoordsToBEV(x_img+box_width*config.OBJ_WIDTH_BEV_SCALING,z)
			max_x = abs(x) if abs(x) > max_x else max_x

			ax4.scatter(x,z,color=clscolors[int(class_pred)],marker='.',s=config.MARKER_SCALING)
			ax4.plot([x1,x2],[z,z],color=clscolors[int(class_pred)],linewidth=0.5)
			if class_pred == 30:
				carRect = patches.Rectangle(
					(x1,z), # car is ~ 4m long
					x2-x1,
					4,
					linewidth=0.7*conf, # plot line thickness as confidence
					edgecolor=clscolors[int(class_pred)],
					facecolor='none',
				)
				ax4.add_patch(carRect)
			
		
			upper_left_x = box[0] - box[2] / 2
			upper_left_y = box[1] - box[3] / 2
			rect = patches.Rectangle(
			(upper_left_x * width, upper_left_y * height),
			box[2] * width,
			box[3] * height,
			linewidth=0.5,
			edgecolor=clscolors[int(class_pred)],
			facecolor="none",
			)
			# Add the patch to the Axes
			ax2.add_patch(rect)
			ax2.text(
			upper_left_x * width,
			upper_left_y * height,
			s="c:" + str(int(class_pred)) + f"d:{box[4]:.1f}" ,
			# s=f"d:{box[4]:.1f}",
			color="white",
			verticalalignment="top",
			bbox={"color": clscolors[int(class_pred)], "pad": 0},

			)
		ax2.set_title(f'Target, ratio of invalid/valid depth : {noOfInvalid}/{len(target_boxes)}')
		if config.FIX_X_LIMIT == True:
			ax3.set_xlim([- config.FIX_BEV_X_LIM,config.FIX_BEV_X_LIM])
			ax4.set_xlim([- config.FIX_BEV_X_LIM,config.FIX_BEV_X_LIM])
		else:
			ax3.set_xlim([-max_x-1,+max_x+1])
			ax4.set_xlim([-max_x-1,+max_x+1])
		ax4.set_ylim([1,100])
		
	if figure==None:
		plt.savefig(filename,dpi=500)

# TODO for plotting dense prediction or calculate dense difference between pred and label
#
		# depth_target = np.array(Image.open(depth_label_path))
		# valid_depth_pixels = depth_target > 0
		# # outputs two vectors (x y) of length n
		# valid_depth_pixels_indices = np.where(valid_depth_pixels == True)
		# # this does output a 1d vector of length n
		# valid_depth_pixels_values = depth_target[valid_depth_pixels_indices]
		# # this might be error
		# grid_x, grid_y = np.mgrid[0:1536, 0:4096]
		# # or this might be wrong: 2 xy vector for indices might need to be transposed or something
		# dense_depth_target = griddata(valid_depth_pixels_indices,valid_depth_pixels_values,(grid_x,grid_y),method='nearest')
		# # dense_depth_target = dense_depth_target.astype(np.dtype('int32'))
		# savePath=os.path.join(self.depth_labels_dir,'griddata_linear',self.annotations.iloc[index, 1])
		# Image.fromarray(dense_depth_target).save(savePath)
		# return 0


# TODO run this function on diff from target and prediction
def sparseDepthPredictionToDenseMap(depth_target,epoch,foldername,all_bboxes_global_coords,nms_pred_boxes_at_scale,nms_all_pred_boxes,target_boxes,figure=None,is_preds=True):
	'''
	### Plots the predicted depth of each bbox at Scale S by extrapolating by nearest neighbour with griddata() \
		to create a dense depth map of the prediction for comparision with the GT Griddata NN labels

	## Input: 
	Prediction Tensor aus model(x)  mit shape [BS,NumOfBBoxPerCell,S,S,num_classes + conf,xywhz], config.ANCHORS in [0..1] 

	## Output:
	np. array 
	look at every bbox: coordinate, which pixel corresponds to the cp 
	assign the value of the prediction to this pixel. take the bbox with highest confidence
	or take the mean of all bboxes corresponding to the pixel value 
	[0.1 .. 255 ] in metern.
	'''

	
	#    cells_to bboxes returned,  converted_bboxes = torch.cat((best_class, scores, x, y, w_h,z), dim=-1).reshape(BATCH_SIZE, num_anchors, S , S, 7)
	def fill_sparse_depth_prediction(global_bboxes_i):
		map=torch.zeros((config.IMAGE_SIZE,config.IMAGE_SIZE),dtype=np.float)
		for box in global_bboxes_i: # iterate over bboxes in img
			clip = lambda x, l, u: max(l, min(u, x))
			x=clip(int(box[2] * config.IMAGE_SIZE),0,config.IMAGE_SIZE-1)
			y=clip(int(box[3] * config.IMAGE_SIZE),0,config.IMAGE_SIZE-1)
			map[x,y]=clip(box[6],0.1,255) # assign depth value of bbox
		img=map.detach().cpu()
		# TODO set cmap to turbo or jet
		# cmap=plt.get_cmap("jet")
		img=np.array(img)

		depth_pred_sparse = img
		valid_depth_pixels = depth_pred_sparse > 0
		# outputs two vectors (x y) of length n
		valid_depth_pixels_indices = np.where(valid_depth_pixels == True)

		

		if(len(valid_depth_pixels_indices[0])>0):
			# this does output a 1d vector of length n
			valid_depth_pixels_values = depth_pred_sparse[valid_depth_pixels_indices]
			# this might be error
			grid_x, grid_y = np.mgrid[0:config.IMAGE_SIZE, 0:config.IMAGE_SIZE]
			# or this might be wrong: 2 xy vector for indices might need to be transposed or something
			dense_depth_pred = griddata(valid_depth_pixels_indices,valid_depth_pixels_values,(grid_x,grid_y),method='nearest')
			dense_depth_pred = np.transpose(dense_depth_pred) # fore some reason it has to be transposed
			return dense_depth_pred

	# dense_depth_target = dense_depth_target.astype(np.dtype('int32'))
	dense_depth_pred2 = fill_sparse_depth_prediction(all_bboxes_global_coords[2]) # all scales nms
	dense_depth_pred1 = fill_sparse_depth_prediction(all_bboxes_global_coords[1])
	dense_depth_pred0 = fill_sparse_depth_prediction(all_bboxes_global_coords[0])




	turbocmap=ListedColormap(turbo_colormap.turbo_colormap_data)
	filename=f'{foldername}/depth_scale?_ep{epoch}.png'
	if figure!=None:
		fig=figure
	else:
		fig = plt.figure(dpi=150)
		
	if(depth_target.dim()>1):
		ax2 = fig.add_subplot(3, 3, 5)
		ax3 = fig.add_subplot(3, 3, 6)
		ax4 = fig.add_subplot(3, 3, 9)
		ax5 = fig.add_subplot(3, 3, 4)
		ax5.set_title('Predictions_filled_bboxes')
		ax5.imshow(np.zeros((config.IMAGE_SIZE,config.IMAGE_SIZE)),cmap=turbocmap,vmin=config.DEPTH_COLORMAP_MIN,vmax=config.DEPTH_COLORMAP_MAX)
		imgplot2 = ax2.imshow(depth_target,cmap=turbocmap,vmin=config.DEPTH_COLORMAP_MIN,vmax=config.DEPTH_COLORMAP_MAX)
		ax2.set_title('Target')
		fig.subplots_adjust(right=0.8)
		cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
		fig.colorbar(imgplot2,cax=cbar_ax)
	else:
		ax2 = fig.add_subplot(3, 3,4)
		ax2.set_title('No 2D Depth Target available')
	ax1 = fig.add_subplot(3, 3, 3)
	imgplot1 = ax1.imshow(dense_depth_pred2,cmap=turbocmap,vmin=config.DEPTH_COLORMAP_MIN,vmax=config.DEPTH_COLORMAP_MAX)
	imgplot3 = ax3.imshow(dense_depth_pred1,cmap=turbocmap,vmin=config.DEPTH_COLORMAP_MIN,vmax=config.DEPTH_COLORMAP_MAX)
	imgplot4 = ax4.imshow(dense_depth_pred0,cmap=turbocmap,vmin=config.DEPTH_COLORMAP_MIN,vmax=config.DEPTH_COLORMAP_MAX)
	ax1.set_title(f'Prediction at Scale 2 ({config.S[2]})')
	ax3.set_title(f'Prediction at Scale 1 ({config.S[1]})')
	ax4.set_title(f'Prediction at Scale 0 ({config.S[0]})')
	
	cmap1 = plt.get_cmap("tab20")
	cmap2 = plt.get_cmap("tab20c")
	colors1 = [cmap1(i) for i in np.linspace(0, 1, 20)]
	colors2 = [cmap2(i) for i in np.linspace(0,1,config.NUM_CLASSES-20)]
	colors1.extend(colors2)
	clscolors=colors1
	
	height, width = config.IMAGE_SIZE,config.IMAGE_SIZE
	nms_all_pred_boxes = sorted(nms_all_pred_boxes, key=lambda x: x[6],reverse=True)
	
	for box in nms_all_pred_boxes:
		class_pred = box[0]
		conf = box[1]
		box = box[2:]
		upper_left_x = box[0] - box[2] / 2
		upper_left_y = box[1] - box[3] / 2
		depth=box[4]
		rect = patches.Rectangle(
			(upper_left_x * width, upper_left_y * height),
			box[2] * width,
			box[3] * height,
			linewidth=0.5 * conf,
			# color=turbo_colormap.interpolate_or_clip(turbo_colormap.turbo_colormap_data,depth/config.TURBO_COLORMAP_MAX),
			edgecolor=clscolors[int(class_pred)],
			facecolor=turbo_colormap.interpolate_or_clip(turbo_colormap.turbo_colormap_data,depth/config.DEPTH_COLORMAP_MAX),
		)
		ax5.add_patch(rect)

		# ax5.text(
		#     upper_left_x * width,
		#     upper_left_y * height,
		#     s="c:" + str(int(class_pred)) + f"d:{box[4]:.1f} cf:{conf:.2f}" ,
		#     # s=f"d:{box[4]:.1f}",
		#     color="white",
		#     verticalalignment="top",
		#     bbox={"color": clscolors[int(class_pred)], "pad": 0},
		# )

	for box in nms_pred_boxes_at_scale[2]:
		assert len(box) == 7, "box should contain class pred, confidence, x, y, width, height and depth"
		class_pred = box[0]
		conf = box[1]
		box = box[2:]
		upper_left_x = box[0] - box[2] / 2
		upper_left_y = box[1] - box[3] / 2
		rect = patches.Rectangle(
			(upper_left_x * width, upper_left_y * height),
			box[2] * width,
			box[3] * height,
			linewidth=0.7*conf, # plot line thickness as confidence
			edgecolor=clscolors[int(class_pred)],
			facecolor="none",
		)
		# Add the patch to the Axes
		ax1.add_patch(rect)
		ax1.text(
			upper_left_x * width,
			upper_left_y * height,
			s="c:" + str(int(class_pred)) + f"d:{box[4]:.1f} cf:{conf:.2f}" ,
			# s=f"d:{box[4]:.1f}",
			color="white",
			verticalalignment="top",
			bbox={"color": clscolors[int(class_pred)], "pad": 0},
		)
	for box in nms_pred_boxes_at_scale[1]:
		assert len(box) == 7, "box should contain class pred, confidence, x, y, width, height and depth"
		class_pred = box[0]
		conf = box[1]
		box = box[2:]
		upper_left_x = box[0] - box[2] / 2
		upper_left_y = box[1] - box[3] / 2
		rect = patches.Rectangle(
			(upper_left_x * width, upper_left_y * height),
			box[2] * width,
			box[3] * height,
			linewidth=0.7*conf, # plot line thickness as confidence
			edgecolor=clscolors[int(class_pred)],
			facecolor="none",
		)
		# Add the patch to the Axes
		ax3.add_patch(rect)
		ax3.text(
			upper_left_x * width,
			upper_left_y * height,
			s="c:" + str(int(class_pred)) + f"d:{box[4]:.1f} cf:{conf:.2f}" ,
			# s=f"d:{box[4]:.1f}",
			color="white",
			verticalalignment="top",
			bbox={"color": clscolors[int(class_pred)], "pad": 0},
		)
	for box in nms_pred_boxes_at_scale[0]:
		assert len(box) == 7, "box should contain class pred, confidence, x, y, width, height and depth"
		class_pred = box[0]
		conf = box[1]
		box = box[2:]
		upper_left_x = box[0] - box[2] / 2
		upper_left_y = box[1] - box[3] / 2
		rect = patches.Rectangle(
			(upper_left_x * width, upper_left_y * height),
			box[2] * width,
			box[3] * height,
			linewidth=0.7*conf, # plot line thickness as confidence
			edgecolor=clscolors[int(class_pred)],
			facecolor="none",
		)
		# Add the patch to the Axes
		ax4.add_patch(rect)
		ax4.text(
			upper_left_x * width,
			upper_left_y * height,
			s="c:" + str(int(class_pred)) + f"d:{box[4]:.1f} cf:{conf:.2f}" ,
			# s=f"d:{box[4]:.1f}",
			color="white",
			verticalalignment="top",
			bbox={"color": clscolors[int(class_pred)], "pad": 0},
		)


	target_boxes =[bbox for bbox in target_boxes if bbox[1]!=0]

	for box in target_boxes:
		assert len(box) == 7, "box should contain class pred, confidence, x, y, width, height and depth"
		class_pred = box[0]
		box = box[2:]
		upper_left_x = box[0] - box[2] / 2
		upper_left_y = box[1] - box[3] / 2
		rect = patches.Rectangle(
			(upper_left_x * width, upper_left_y * height),
			box[2] * width,
			box[3] * height,
			linewidth=0.5,
			edgecolor=clscolors[int(class_pred)],
			facecolor="none",
		)
		# Add the patch to the Axes
		ax2.add_patch(rect)
		# ax2.text(
		#     upper_left_x * width,
		#     upper_left_y * height,
		#     s="c" + str(int(class_pred)) + f" d{box[4]:.1f}" ,
		#     # s=f"d:{box[4]:.1f}",
		#     color="white",
		#     verticalalignment="top",
		#     bbox={"color": clscolors[int(class_pred)], "pad": 0},

		# )

	if(figure==None):
		plt.savefig(filename)
		plt.clf()

def plot_couple_examples(model, loader:DataLoader, conf_threshold, iou_threshold, anchors,depth_anchors,noOfExamples,epochNo=0,datasetName=''):
	model.eval()
	noOfBatchesToPlot = math.ceil(noOfExamples/config.BATCH_SIZE)
	tempNoOfWorkers = loader.num_workers
	loader.num_workers = 1
	for batchidx, (x, labels, depth_target, img_filename) in tqdm(enumerate(loader)):
		if batchidx >= noOfBatchesToPlot:
			break
		x = x.to(config.DEVICE,dtype=torch.float)
		all_pred_boxes = []
		all_true_boxes = []
		with torch.no_grad():
			predictions = model(x)
			fig=plt.figure(figsize = (15,15))
			batch_size = x.shape[0]
			bboxes = [[] for _ in range(batch_size)]
			list_bboxes_at_scale=[]
			for i in range(3):
				S = predictions[i].shape[2]
				scaled_anchor_at_S = anchors[i]
				depth_anchors_at_S = depth_anchors[i]
				# list [100bbox,400bbox,1000bbox]

				boxes_scale_i = cells_to_bboxes(
					predictions[i], scaled_anchor_at_S,depth_anchors=depth_anchors_at_S, S=S, is_preds=True
				).tolist()
				list_bboxes_at_scale.append(boxes_scale_i)
				for idx, (box) in enumerate(boxes_scale_i):
					bboxes[idx] += box
			# we just want one bbox for each label, not one for each scale


			true_bboxes = cells_to_bboxes(
				labels[2], anchors[2], S=predictions[2].shape[2], is_preds=False
			).tolist()
			plt.rc('font',size=2) 
			plt.rc('axes',titlesize=6)
			plt.rc('xtick',labelsize =5)
			plt.rc('ytick',labelsize =5)
			for idx in trange(min(noOfExamples,batch_size),desc='Plot Examples: '+datasetName, leave=True):
				# take only bboxes prediction with valid depth prediction:
				bboxes_valid = [bbox for bbox in bboxes[idx] if bbox[6]>1] # depth must be greater than 1
				nms_all_boxes = nms_torch(
					bboxes_valid,
					iou_threshold=iou_threshold,
					threshold=conf_threshold,
					# box_format=box_format,
				)

				list_bboxes_at_scale_idx=[list_bboxes_at_scale[i][idx] for i in range(3)]
				list_nms_bboxes_at_scale=[nms_torch(list_bboxes_at_scale_idx[i],iou_threshold,conf_threshold) for i in range(3)]
				fn = f'{config.TRAINING_LOG_DIR}/{epochNo}_{datasetName}_{batchidx*config.BATCH_SIZE+idx}_{img_filename[idx]}.png'
				# plot the depth predictions with bbox detection
				# TODO Idea plot conditional class distribution in the same fashion
				sparseDepthPredictionToDenseMap(depth_target[idx], epochNo, config.TRAINING_EXAMPLES_PLOT_DIR_DEPTH,
												list_bboxes_at_scale_idx, list_nms_bboxes_at_scale,nms_all_boxes, true_bboxes[idx],
												fig)
				# plot the bbox detection in rgb image
				plot_image(x[idx].permute(1,2,0).detach().cpu(), nms_all_boxes,filename=fn,figure=fig,target_boxes=true_bboxes[idx],datasetName=datasetName)
				plt,plt.suptitle(img_filename[idx])
				plt.savefig(fn,dpi=300)
				plt.clf()

				
				
	loader.num_workers=tempNoOfWorkers
	model.train()


def plotDepthPrediction(densePredictionArray):
	'''
	Input: Numpy Array
	Output: no output, save img to disk 
	'''
	pass

def plotDifferencePredictionLabel(DepthPredictionArray, TargetArray):
	'''
	Input: Prediction, Label
	Output: plot colourcoded difference, save img, with colormap from 0.1 to 255
	'''

def evaluateDepthPrediction(prediction, target):
	'''
	Input: Depth Prediction, target
	Output: metric that tells an interpretable value of miss prediction like accuracy etc
	'''
	pass

def evaluateDepthPredictionObjectsOnly(prediction, target):
	'''
	Input: prediction, target
	Output: metric taking only the objects of the target into account. similiar to the loss that only accounts for objects.
	'''
def get_evaluation_bboxes(
	loader,
	model,
	iou_threshold,
	anchors,
	depth_anchors,
	threshold,
	box_format="midpoint",
	device="cuda",
):
	# make sure model is in eval before get bboxes
	model.eval()
	train_idx = 0
	all_pred_boxes = []
	all_true_boxes = []
	train_idx_filename_dict = {}
	for batch_idx, (x, labels, _, filenames) in enumerate(tqdm(loader)):
		x = x.to(config.DEVICE,dtype=torch.float)

		with torch.no_grad():
			predictions = model(x)

		batch_size = x.shape[0]
		bboxes = [[] for _ in range(batch_size)]
		for i in range(3):
			S = predictions[i].shape[2]
			anchor = torch.tensor([*anchors[i]]).to(config.DEVICE) * S
			boxes_scale_i = cells_to_bboxes(
				predictions[i], anchor,depth_anchors=depth_anchors[i], S=S, is_preds=True
			).tolist()
			for idx, (box) in enumerate(boxes_scale_i):
				bboxes[idx] += box

		# we just want one bbox for each label, not one for each scale
		true_bboxes = cells_to_bboxes(
			labels[2], anchor, S=S, is_preds=False
		).tolist()

		for idx in range(batch_size):
			nms_boxes = nms_torch(
				bboxes[idx],
				iou_threshold=iou_threshold,
				threshold=threshold,
				# box_format=box_format,
			)
			# TODO append filename, plot boxes to image, mark TP Box Predictions with depth error
			for nms_box in nms_boxes:
				all_pred_boxes.append([train_idx] + nms_box)

			for box in true_bboxes[idx]:
				if box[1] > threshold:
					all_true_boxes.append([train_idx] + box)
			train_idx_filename_dict[train_idx]=filenames[idx]
			train_idx += 1

	model.train()
	return all_pred_boxes, all_true_boxes, train_idx_filename_dict

# TODO also return depth predictions with bbox
def cells_to_bboxes(predictions, anchors, S,depth_anchors=None,is_preds=True,reshape=True):
	"""
	when ispred=False, no depth_anchors are needed

	Scales the predictions coming from the model to
	be relative to the entire image such that they for example later
	can be plotted or.
	INPUT:
	predictions: tensor of size (N, 3, S, S, num_classes+5)
	anchors: the anchors used for the predictions
	S: the number of cells the image is divided in on the width (and height)
	is_preds: whether the input is predictions or the true bounding boxes
	OUTPUT:
	converted_bboxes: the converted boxes of sizes (N, num_anchors, S, S, 1+5) with class index,
					  object score, bounding box coordinates
	"""
	BATCH_SIZE = predictions.shape[0]
	num_anchors = len(anchors)
	box_predictions = predictions[..., 1:5]
	if is_preds:
		anchors = anchors.reshape(1, len(anchors), 1, 1, 2)
		box_predictions[..., 0:2] = torch.sigmoid(box_predictions[..., 0:2])
		box_predictions[..., 2:] = anchors * torch.exp(box_predictions[..., 2:])
		scores = torch.sigmoid(predictions[..., 0:1])
		best_class = torch.argmax(predictions[..., 6:], dim=-1).unsqueeze(-1)
		
		depth_anchors = depth_anchors.reshape(1,len(depth_anchors),1,1,1)
		depth = depth_anchors * torch.exp(predictions[...,5:6])
	else:
		scores = predictions[..., 0:1]
		best_class = predictions[..., 6:7]
		depth = predictions[...,5:6]

	cell_indices = (
		torch.arange(S)
		.repeat(predictions.shape[0], 3, S, 1)
		.unsqueeze(-1)
		.to(predictions.device)
	)
	x = 1 / S * (box_predictions[..., 0:1] + cell_indices)
	y = 1 / S * (box_predictions[..., 1:2] + cell_indices.permute(0, 1, 3, 2, 4))
	w_h = 1 / S * box_predictions[..., 2:4]
	z = depth
	if reshape:
		converted_bboxes = torch.cat((best_class, scores, x, y, w_h,z), dim=-1).reshape(BATCH_SIZE, num_anchors * S * S, 7)
	else:
		converted_bboxes = torch.cat((best_class, scores, x, y, w_h,z), dim=-1).reshape(BATCH_SIZE, num_anchors, S , S, 7)

	return converted_bboxes

def check_class_accuracy(model, loader, threshold):
	model.eval()
	tot_class_preds, correct_class = 0, 0
	tot_noobj, correct_noobj = 0, 0
	tot_obj, correct_obj = 0, 0

	for idx, (x, y,_,_) in enumerate(tqdm(loader)):
		x = x.to(config.DEVICE,dtype=torch.float)
		with torch.no_grad():
			out = model(x)

		for i in range(3):
			y[i] = y[i].to(config.DEVICE)
			obj = y[i][..., 0] == 1 # in paper this is Iobj_i
			noobj = y[i][..., 0] == 0  # in paper this is Iobj_i

			correct_class += torch.sum(
				torch.argmax(out[i][..., 6:][obj], dim=-1) == y[i][..., 6][obj]
			)
			tot_class_preds += torch.sum(obj)

			obj_preds = torch.sigmoid(out[i][..., 0]) > threshold
			correct_obj += torch.sum(obj_preds[obj] == y[i][..., 0][obj])
			tot_obj += torch.sum(obj)
			correct_noobj += torch.sum(obj_preds[noobj] == y[i][..., 0][noobj])
			tot_noobj += torch.sum(noobj)
	clsAccuracy = (correct_class/(tot_class_preds+1e-16))*100
	noobjAccuracy =(correct_noobj/(tot_noobj+1e-16))*100
	objAccuracy = (correct_obj/(tot_obj+1e-16))*100
	print(f"Class accuracy is: {clsAccuracy:2f}%")
	print(f"No obj accuracy is: {noobjAccuracy:2f}%")
	print(f"Obj accuracy is: {objAccuracy:2f}%")
	model.train()
	return clsAccuracy,noobjAccuracy,objAccuracy


def get_mean_std(loader):
	# var[X] = E[X**2] - E[X]**2
	channels_sum, channels_sqrd_sum, num_batches = 0, 0, 0

	for data, _,_ in tqdm(loader):
		channels_sum += torch.mean(data, dim=[0, 2, 3])
		channels_sqrd_sum += torch.mean(data ** 2, dim=[0, 2, 3])
		num_batches += 1

	mean = channels_sum / num_batches
	std = (channels_sqrd_sum / num_batches - mean ** 2) ** 0.5

	return mean, std


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
	print("=> Saving checkpoint")
	checkpoint = {
		"state_dict": model.state_dict(),
		"optimizer": optimizer.state_dict(),
	}
	torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
	print("=> Loading checkpoint")
	shutil.copyfile(checkpoint_file,checkpoint_file+'_pre_loading_backup', follow_symlinks=True)

	checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
	model.load_state_dict(checkpoint["state_dict"])
	optimizer.load_state_dict(checkpoint["optimizer"])

	# If we don't do this then it will just have learning rate of old checkpoint
	# and it will lead to many hours of debugging \:
	for param_group in optimizer.param_groups:
		param_group["lr"] = lr

def load_checkpoint_transfer(checkpoint_file, model, optimizer, lr):
	print("=> Loading checkpoint")
	shutil.copyfile(checkpoint_file,'pre_loading_backup_'+checkpoint_file, follow_symlinks=True)
	pretrained_dict = torch.load(checkpoint_file, map_location=config.DEVICE)
	# model_dict = model.state_dict()
	pretrained_dict_state = pretrained_dict["state_dict"]
	# # 1. filter out final detection layers
	pretrained_dict_state = {k: v for k, v in pretrained_dict_state.items() if 'layers.15.pred.1' not in k}
	pretrained_dict_state = {k: v for k, v in pretrained_dict_state.items() if 'layers.22.pred.1' not in k}
	pretrained_dict_state = {k: v for k, v in pretrained_dict_state.items() if 'layers.29.pred.1' not in k}
	# # 2. overwrite entries in the existing state dict
	# model_dict.update(pretrained_dict) 
	# # 3. load the new state dict
	# model.load_state_dict(pretrained_dict)
	model.load_state_dict(pretrained_dict_state,strict=False)
	# We probably dont want to take the checkpoint optimizer params when transfer learning 

	# optimizer.load_state_dict(pretrained_dict["optimizer"])

	# # If we don't do this then it will just have learning rate of old checkpoint
	# # and it will lead to many hours of debugging \:
	# for param_group in optimizer.param_groups:
	#     param_group["lr"] = lr




def seed_everything(seed=42):
	os.environ['PYTHONHASHSEED'] = str(seed)
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
