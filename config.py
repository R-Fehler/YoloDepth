import albumentations as A
import cv2
import torch

from albumentations.pytorch import ToTensorV2
from utils import seed_everything
DS_NAME='Ostring'
# DS_NAME='OstringOverfit'
# DS_NAME='Mapillary'

    # Ostring
    # --------
if DS_NAME=='Ostring':
    # DATASET = 'DepthAndObjectDetection'
    DATASET = 'ObjectDetection'
    DATASET_TRAIN_CSV = 'OstringDepthDataset/OstringDataSetTrainingAndVal.csv'
    DATASET_VAL_CSV = 'OstringDepthDataset/OstringDataSetValidation.csv'
    DATASET_TEST_CSV = 'OstringDepthDataset/OstringDataSetTest.csv'
    IMAGE_DIR = 'OstringDepthDataset/imgs/'
    BBOX_LABEL_DIR = "OstringDepthDataset/bbox_labels/"
    DEPTH_NN_MAP_LABEL = "OstringDepthDataset/depth_labels/griddata_nearest/"
    NUM_CLASSES = 80


    # Ostring low Examples Overfit
    # --------
if DS_NAME=='OstringOverfit':
    DATASET = 'DepthAndObjectDetection'
    DATASET_TRAIN_CSV = 'OstringDepthDataset/OstringDataSet_1_example.csv'
    DATASET_VAL_CSV = 'OstringDepthDataset/OstringDataSet_1_example.csv'
    DATASET_TEST_CSV = 'OstringDepthDataset/OstringDataSet_1_example.csv'
    IMAGE_DIR = 'OstringDepthDataset/imgs/'
    BBOX_LABEL_DIR = "OstringDepthDataset/bbox_labels/"
    DEPTH_NN_MAP_LABEL = "OstringDepthDataset/depth_labels/griddata_nearest/"
    NUM_CLASSES = 80

# Mapillary Vistas
# ----------------
if DS_NAME == 'Mapillary':
    DATASET = 'ObjectDetection'
    DATASET_TRAIN_CSV = 'MapillaryVistasDataset/Mapillary_Vistas_Training.csv'
    DATASET_VAL_CSV = 'MapillaryVistasDataset/Mapillary_Vistas_Validation.csv'
    DATASET_TEST_CSV = 'MapillaryVistasDataset/Mapillary_Vistas_Test.csv'
    IMAGE_DIR = 'MapillaryVistasDataset/images/total'
    BBOX_LABEL_DIR = "MapillaryVistasDataset/labels/total"
    NUM_CLASSES = 37
# DEPTH_NN_MAP_LABEL = "OstringDepthDataset/depth_labels/griddata_nearest/"


# DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DEVICE = "cuda:4"
# DEVICE = "cpu"
# seed_everything()  # If you want deterministic behavior
NUM_WORKERS = 96
BATCH_SIZE = 36
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 2000
CONF_THRESHOLD = 0.1
MAP_IOU_THRESH = 0.5
NMS_IOU_THRESH = 0.45
IMAGE_SIZE = 416
S = [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8]
DEPTH_MASK_THRESHOLD = 1.0
PIN_MEMORY = True
LOAD_MODEL = True
TRANSFER_MODEL = False # because we dont change detection heads
SAVE_MODEL = True
LOAD_CHECKPOINT_FILE = "yolov3_pascal_78.1map_saved_correct_cls_labels_correct_anchors_Ostring_Detection_Only.pth.tar"
SAVE_CHECKPOINT_FILE = "yolov3_pascal_78.1map_saved_correct_cls_labels_correct_anchors_Ostring_Detection_max_obj_test.pth.tar"
IMG_DIR = DATASET + "/images/"
LABEL_DIR = DATASET + "/labels/"
TRAINING_EXAMPLES_PLOT_DIR = 'imgs_Ostring_testing_max_obj_no_depth'
TRAINING_EXAMPLES_PLOT_DIR_DEPTH = TRAINING_EXAMPLES_PLOT_DIR +'/depthPred'
# ANCHORS = [
#     [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
#     [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
#     [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
# ]  # Note these have been rescaled to be between [0, 1]

ANCHORS = [
    [(0.04375, 0.03984375), (0.0296875, 0.13203125), (0.153125, 0.1203125)],
    [(0.009375, 0.01953125), (0.021875, 0.01640625), (0.0109375, 0.06328125)],
    [(0.00390625, 0.00390625), (0.00234375, 0.01953125), (0.0078125, 0.0078125)],
]

scale = 1.1
train_transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=int(IMAGE_SIZE * scale)),
        A.PadIfNeeded(
            min_height=int(IMAGE_SIZE * scale),
            min_width=int(IMAGE_SIZE * scale),
            border_mode=cv2.BORDER_CONSTANT,
        ),
        A.RandomCrop(width=IMAGE_SIZE, height=IMAGE_SIZE),
        A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=0.4),
        A.OneOf(
            [
                A.ShiftScaleRotate(
                    rotate_limit=20, p=0.5, border_mode=cv2.BORDER_CONSTANT
                ),
                A.IAAAffine(shear=15, p=0.5, mode="constant"),
            ],
            p=1.0,
        ),
        A.HorizontalFlip(p=0.5),
        A.Blur(p=0.1),
        A.CLAHE(p=0.1),
        A.Posterize(p=0.1),
        A.ToGray(p=0.1),
        A.ChannelShuffle(p=0.05),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[],),
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

MY_TRANSFORMS = A.Compose(
    [
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
        ToTensorV2(),
    ]
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
