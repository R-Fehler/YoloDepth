import albumentations as A
import cv2
import torch
import torch.cuda

from albumentations.pytorch import ToTensorV2
from utils import seed_everything
from collections import namedtuple

DEVICE = "cuda:1" if torch.cuda.is_available() else 'cpu' # select the device by cuda:0 or cuda:1 etc, check gpustat in terminal
TRAINING_LOG_DIR = 'runs_mrtstorage/train/mAP_Depth_Evaluation_Plotting_Outliers/'
LOAD_CHECKPOINT_FILE = "runs_mrtstorage/train/from_PascalVOC_416px_ObjDepth_correct_depth_targets_transfer_toCompare_1/yolov3_pascal_78.1map_Dual_Dataset_Obj_specific_Depth_anchorDepths_batch_split_transfer_from_PASCALVOC_416px_only_obj_depth_loss_correct_targets.pth.tarOSTbestMAP_0.07581827789545059"
# LOAD_CHECKPOINT_FILE = "runs_mrtstorage/train/from_PascalVOC_416px_only_ObjDepth_savingBestModel/yolov3_pascal_78.1map_Dual_Dataset_Obj_specific_Depth_anchorDepths_batch_split_transfer_from_PASCALVOC_416px_only_obj_depth_loss.pth.tarbestOST"
SAVE_CHECKPOINT_FILE = TRAINING_LOG_DIR + "yolov3_pascal_78.1map_Dual_Dataset_Obj_specific_Depth_anchorDepths_batch_split_transfer_from_PASCALVOC_416px_NO_obj_depth_loss_correct_targets.pth.tar"
TRANSFER_MODEL = False # when changing num_classes from different datasets, deletes last layer
LOAD_MODEL = True # load weights from SAVE_CHECKPOINT_FILE, False when training from scratch
SAVE_MODEL = False # checkpoint file is saved under SAVE_CHECKPOINT_FILE
SAVE_SRC_AND_CONFIG = True # copies the python src and this config file to training log dir, turn off if you are disturbed by it, helps reproducability in fast changing codebase

# When you just want to plot via the Training Loop, but dont want to train. Select NO_OF_EXAMPLES_TEST etc appropriately and NUM_EPOCH = 1 
JUST_VIZUALIZE_NO_TRAINING = True
OBJ_WIDTH_BEV_SCALING = 1.5 # scaling of width projections of objects in Birds Eye View Plot
MARKER_SCALING = 3
FIX_X_LIMIT = True
FIX_BEV_X_LIM = 30 # width of BEV Plot when looking at sequences of imgs, SHUFFLE_DATA_LOADER must be False otherwise auto scaling
PLOT_COUPLE_EXAMPLES = True # plot after each epoch the defined num of example in and outputs
PLOT_BEFORE_TRAINING = False
NUM_CLASSES = 37

GENERATE_PSEUDO_DEPTH_TARGETS = True # set True when you want to use 1/10 quantile of bbox Region of Interest as depth of given object in bbox (pseudolabels on ostring dataset)


QUANTILE_DEPTH_CLUSTER =0.12 # take a lidar value close to the camera (eg.0.0-0.5)

USE_DATA_AUGMENTATION_TRAIN = False # still buggy, bboxes are not returned from albumentation transform

NO_OF_EXAMPLES_TEST = 1 # no of examples from test datasets  (each)
NO_OF_EXAMPLES_TRAIN = 1 #  no of examples from train datasets  (each)
DEPTH_COLORMAP_MIN = 0 # min value of plot colormap in meter
DEPTH_COLORMAP_MAX = 120 # max value of plot colormap in meter, everything above this has same color as max

EVAL_DEPTH_ERROR=True
MIN_DEPTH_EVAL = 10
MAX_DEPTH_EVAL = 100
PLOT_EVAL_ERROR=True
PLOT_EVAL_ERROR_THRESHOLD = 1 # relative error smaller this will be plotted for investigation
EVAL_PLOTS_DIR='yoloDepth_eval_above_100_relative_error/'
EVAL_FILE_NAME='eval_depth_error_ost_test.csv'


PLOT_CAR_BOXES_NON_LOG_SPACE = True # plots cars as 4m long boxes in BEV and switches logarithmic scale off

NUM_EPOCHS = 1
SHUFFLE_DATA_LOADER = True

MAX_NO_OF_DETECTIONS = 200 # limits number of bboxes that are processed by classwise non max supression

# Only for Ostring Dataset
ORIGINAL_IMAGE_WIDTH = 4096
ORIGINAL_IMAGE_HEIGHT = 1536
IMAGE_SIZE = 416
S = [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8] # when you change this, you need to change the model architecture
# S = [IMAGE_SIZE // 64, IMAGE_SIZE // 32, IMAGE_SIZE // 16] 

LABEL_WIDTH_THRESHOLD = 2/IMAGE_SIZE # smallest bbox target width and height in pixel (2 pixel here)
LABEL_HEIGHT_THRESHOLD = 2/IMAGE_SIZE
# C-like struct datastructure for easy DATASET config and access in main
DS_CONFIG=namedtuple('DS_CONFIG','DATASET DATASET_TRAIN_CSV DATASET_VAL_CSV  DATASET_TEST_CSV IMAGE_DIR BBOX_LABEL_DIR DEPTH_NN_MAP_LABEL NUM_CLASSES BOX_LABEL_COL' )
USE_2D_DEPTH_MAP = True

#=============================================
# Flag for activating Obj specific depth loss
#=============================================
USE_OBJ_DEPTH_LOSS = True

USE_NO_OBJ_DEPTH_LOSS = False # set true when you want to incorporate the depth loss at every grid cell for every anchor box and scale (dense depth pred), much worse generalization to MVD Depth Prediction
# Datasets need a directory with all imgs, a dir with all label.txt files, a dir with all preprocessed depth map imgs,
# Then you provide a csv file that contain the filenames of the files that are contained in the given data split
# eg. train.csv, test.csv, val.csv


    # Ostring
    # --------------------------
OSTRING_DS = DS_CONFIG(
    DATASET = 'DepthAndObjectDetection',
    # DATASET = 'ObjectDetection',
    DATASET_TRAIN_CSV = 'OstringDepthDataset/OstringDataSetTrainingAndVal_Mapping.csv',
    DATASET_VAL_CSV = 'OstringDepthDataset/OstringDataSetValidation.csv',
    DATASET_TEST_CSV = 'OstringDepthDataset/OstringDataSetTest_MappingNOTGT.csv',
    IMAGE_DIR = 'OstringDepthDataset/imgs/',
    BBOX_LABEL_DIR = "OstringDepthDataset/bbox_labels/",
    DEPTH_NN_MAP_LABEL = "OstringDepthDataset/depth_labels/griddata_nearest_wo_sky/" if USE_2D_DEPTH_MAP else None,
    NUM_CLASSES = NUM_CLASSES,
    BOX_LABEL_COL = 2
    )


    # Ostring low Examples Overfit
    # -----------------------------
OSTRING_OVERFIT_DS=DS_CONFIG(
    DATASET = 'DepthAndObjectDetection',
    DATASET_TRAIN_CSV = 'OstringDepthDataset/OstringDataSet_1_example.csv',
    DATASET_VAL_CSV = 'OstringDepthDataset/OstringDataSet_1_example.csv',
    DATASET_TEST_CSV = 'OstringDepthDataset/OstringDataSet_1_example.csv',
    IMAGE_DIR = 'OstringDepthDataset/imgs/',
    BBOX_LABEL_DIR = "OstringDepthDataset/bbox_labels/",
    DEPTH_NN_MAP_LABEL = "OstringDepthDataset/depth_labels/griddata_nearest_wo_sky/" if USE_2D_DEPTH_MAP else None,
    NUM_CLASSES = NUM_CLASSES,
    BOX_LABEL_COL = 2
    )

    # Mapillary Vistas
    # ----------------
MAPILLARY_DS=DS_CONFIG(
    DATASET = 'ObjectDetection',
    DATASET_TRAIN_CSV = 'MapillaryVistasDataset/Mapillary_Vistas_Training.csv',
    DATASET_VAL_CSV = 'MapillaryVistasDataset/Mapillary_Vistas_Validation.csv',
    DATASET_TEST_CSV = 'MapillaryVistasDataset/Mapillary_Vistas_Test.csv',
    IMAGE_DIR = 'MapillaryVistasDataset/images/total',
    BBOX_LABEL_DIR = "MapillaryVistasDataset/labels/total",
    DEPTH_NN_MAP_LABEL = None,
    NUM_CLASSES = NUM_CLASSES,
    BOX_LABEL_COL = 1
)


# seed_everything()  # If you want deterministic behavior, under utils, missing in main
NUM_WORKERS = 128 # workers of dataloaders, running not all the time so set it relatively high compared to core count

# 4/1 mvd/ostring ratio worked well, 1/4 mvd/ostring doesnt generalizes depth prediction to mvd images
# ran previous experiments with 30 OST and 6 MVD by mistake, thats why i know :D
# BATCH_SIZE_MVD = 6 
# BATCH_SIZE_OST = 2
BATCH_SIZE_MVD = 28
BATCH_SIZE_OST = 6
BATCH_SIZE = BATCH_SIZE_MVD+BATCH_SIZE_OST # total batch size, max out gpu vram when evaluating test loss
# LEARNING_RATE = 0.2e-3
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
TEST_CONF_THRESHOLD = 0.2 # conf when 
INF_CONF_THRESHOLD = 0.18 # conf threshold when inferring for non max suppression and plotting
MAP_IOU_THRESH = 0.5 # IoU threshold for mean Average Precision, 
NMS_IOU_THRESH_CAR = 0.4 # Idea take seperate IoU threshold for cars because they can be close to each other
NMS_IOU_THRESH = 0.4 # IoU threshold for non max supression, eg when bboxes overlap more than this threshold and one has higher conf. 
                        # low values misses parking cars for example, high values detect the same object multiple times with slightly different bbox
USE_SOFT_NMS = False # experimental and buggy

DIST_TRAFO_THRESHOLD = IMAGE_SIZE//16 # set radius for preprocessing extremely sparse lidar depth projection images (uint 16 bit gray value imgs from Parametric Estimation Tool)
PIN_MEMORY = True
# TRAINING_EXAMPLES_PLOT_DIR = 'imgs_Ostring_OBJ_Depth_LowLR_noOBJ_DepthLoss2'
TRAINING_EXAMPLES_PLOT_DIR_DEPTH = TRAINING_LOG_DIR +'/depthPred'
TB_SUB_DIR = TRAINING_LOG_DIR+'/0_tb_log'
SRC_CODE_BACKUP = TRAINING_LOG_DIR + '/srcCode/'



EVALUATE_TEST_LOSS = False # set to false when you just want to train for testing stuff
EVAL_MAP_FREQ = 2 # set every n-th epoch at which you want to eval mAP, the computation is slow, so small datasets might want high values here
TEST_MAP_AT_START = True # skip first mAP computation?

# Only Idea:
NO_PSEUDO_LABELS_IN_DEPTH_DS = False # WARNING, LOSS NOT IMPLEMENTED FULLY! Idea: let the detector train until it detects with high mAP and conf. on test set then introduce the loss on the depth of the detected bboxes in the depth dataset
UNSUPERVISED_OBJ_THRESHOLD = 0.4 # the threshold for the above idea

# PASCAL VOC
# ANCHORS = [
#     [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
#     [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
#     [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
# ]  # Note these have been rescaled to be between [0, 1]

## ANCHORS FROM MVD AUTOANCHORS yolov5

# ANCHORS = [
#     [(0.04375, 0.03984375), (0.0296875, 0.13203125), (0.153125, 0.1203125)],
#     [(0.009375, 0.01953125), (0.021875, 0.01640625), (0.0109375, 0.06328125)],
#     [(0.00390625, 0.00390625), (0.00234375, 0.01953125), (0.0078125, 0.0078125)],
# ]

# AUTOANCHORS yolov5 on Ostring Dataset
# ANCHORS = [
#     [(0.0625, 0.057692307692307696), (0.06490384615384616, 0.21875), (0.6947115384615384, 0.07932692307692307)],
#     [(0.01201923076923077, 0.026442307692307692), (0.03125, 0.02403846153846154), (0.014423076923076924, 0.08413461538461539)],
#     [(0.007211538461538462, 0.004807692307692308), (0.004807692307692308, 0.019230769230769232), (0.014423076923076924, 0.01201923076923077)],
# ]

# handpicked anchors, choose anchors that are at different scales. here calculated from base aspect ratios
BASE_ANCHORS = [(0.1, 0.1), (0.1, 0.3), (0.7, 0.2)]
BASE_DEPTH_ANCHORS = [10,40,80]
ANCHORS = [BASE_ANCHORS,[ (w/4,h/4) for w,h in BASE_ANCHORS],[ (w/8,h/8) for w,h in BASE_ANCHORS]]


DEPTH_ANCHORS = [
    [BASE_DEPTH_ANCHORS[0] for _ in range(3)],
    [BASE_DEPTH_ANCHORS[1] for _ in range(3)],
    [BASE_DEPTH_ANCHORS[2] for _ in range(3)],
]

TRAINING_IGNORE_IOU_TRESHOLD = 0.5 # Threshold for selecting a anchor box prior as responsible for detecting the target. below this the grid cell target is not responsible (-1 value is set for obj)

scale = 1.1

# transforms are still buggy, bboxes not returning
train_transforms_img = A.Compose(
    [
        # A.LongestMaxSize(max_size=int(IMAGE_SIZE * scale)),
        # A.PadIfNeeded(
        #     min_height=int(IMAGE_SIZE * scale),
        #     min_width=int(IMAGE_SIZE * scale),
        #     border_mode=cv2.BORDER_CONSTANT,
        # ),
        # A.RandomCrop(width=IMAGE_SIZE, height=IMAGE_SIZE),
        # A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=0.4),
        A.OneOf(
            [
                A.ShiftScaleRotate(
                    rotate_limit=20, p=0.5, border_mode=cv2.BORDER_CONSTANT
                ),
                A.IAAAffine(shear=15, p=0.5, mode="constant"),
            ],
            p=1.0,
        ),
        # A.IAAAffine(shear=15, p=0.5, mode="constant"),
        A.HorizontalFlip(p=0.5),
        # A.Blur(p=0.1),
        # A.CLAHE(p=0.1),
        # A.Posterize(p=0.1),
        # A.ToGray(p=0.1),
        # A.ChannelShuffle(p=0.05),
        # A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
        # ToTensorV2(),
    ],
    # changing to coco before transformation and then back to yolo format. yolo was buggy
    bbox_params=A.BboxParams(format="coco", label_fields=[],),
    additional_targets={'depth_target':'image'}
)
test_transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=IMAGE_SIZE),
        A.PadIfNeeded(
            min_height=IMAGE_SIZE, min_width=IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT
        ),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[]),
)


PASCAL_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor"
]

COCO_LABELS = ['person',
 'bicycle',
 'car',
 'motorcycle',
 'airplane',
 'bus',
 'train',
 'truck',
 'boat',
 'traffic light',
 'fire hydrant',
 'stop sign',
 'parking meter',
 'bench',
 'bird',
 'cat',
 'dog',
 'horse',
 'sheep',
 'cow',
 'elephant',
 'bear',
 'zebra',
 'giraffe',
 'backpack',
 'umbrella',
 'handbag',
 'tie',
 'suitcase',
 'frisbee',
 'skis',
 'snowboard',
 'sports ball',
 'kite',
 'baseball bat',
 'baseball glove',
 'skateboard',
 'surfboard',
 'tennis racket',
 'bottle',
 'wine glass',
 'cup',
 'fork',
 'knife',
 'spoon',
 'bowl',
 'banana',
 'apple',
 'sandwich',
 'orange',
 'broccoli',
 'carrot',
 'hot dog',
 'pizza',
 'donut',
 'cake',
 'chair',
 'couch',
 'potted plant',
 'bed',
 'dining table',
 'toilet',
 'tv',
 'laptop',
 'mouse',
 'remote',
 'keyboard',
 'cell phone',
 'microwave',
 'oven',
 'toaster',
 'sink',
 'refrigerator',
 'book',
 'clock',
 'vase',
 'scissors',
 'teddy bear',
 'hair drier',
 'toothbrush'
]

MVD_CLASSES = [
                "Bird",
                "Ground_Animal",
                "Crosswalk_Plain",
                "Person",
                "Bicyclist",
                "Motorcyclist",
                "Other_Rider",
                "Lane_Marking_-_Crosswalk",
                "Banner",
                "Bench",
                "Bike_Rack",
                "Billboard",
                "Catch_Basin",
                "CCTV_Camera",
                "Fire_Hydrant",
                "Junction_Box",
                "Mailbox",
                "Manhole",
                "Phone_Booth",
                "Street_Light",
                "Pole",
                "Traffic_Sign_Frame",
                "Utility_Pole",
                "Traffic_Light",
                "Traffic_Sign_(Back)",
                "Traffic_Sign_(Front)",
                "Trash_Can",
                "Bicycle",
                "Boat",
                "Bus",
                "Car",
                "Caravan",
                "Motorcycle",
                "Other_Vehicle",
                "Trailer",
                "Truck",
                "Wheeled_Slow",
]
