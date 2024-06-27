#!/usr/bin/encv python

# Pallet Segmentation using Detectron2 framework
# Detectron2: https://github.com/facebookresearch/detectron

# Check cuda and torch versions
import torch, detectron2
# get_ipython().system('nvcc --version')
# TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
# CUDA_VERSION = torch.__version__.split("+")[-1]
# print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
# print("detectron2:", detectron2.__version__)


# Some basic setup:
import detectron2
import os, random
import cv2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.data import detection_utils as utils
import detectron2.data
import matplotlib.pyplot as plt
import yaml
import pandas as pd


""" 
Data registration for detectron2 
"""


def data_registration(train_name, valid_name,
                      path_train_coco,path_train_imgs,
                      path_valid_coco,path_valid_imgs):
    """
    Register COCO instances for the training and validation datasets and load the data.

    Args:
        train_name (str): Name for the training dataset.
        valid_name (str): Name for the validation dataset.

    Returns:
        train_data (list): Training data.
        meta_train (MetadataCatalog): Metadata for the training dataset.
        valid_data (list): Validation data.
        meta_valid (MetadataCatalog): Metadata for the validation dataset.
    """
    # path_train_coco = "/home/ai-infinium/Desktop/Dataset/all_data/train.json"
    # path_train_imgs = "/home/ai-infinium/Desktop/Dataset/all_data/images"
    
    # path_valid_coco = "/home/ai-infinium/Desktop/Dataset/all_data/test.json"
    # path_valid_imgs = "/home/ai-infinium/Desktop/Dataset/all_data/images"

    print("Total Train Images in folder => ", len(os.listdir(path_train_imgs)))
    

    print("Total Valid Images in folder =>", len(os.listdir(path_valid_imgs)))

    try:
        register_coco_instances(train_name, {}, path_train_coco, path_train_imgs)
        register_coco_instances(valid_name, {}, path_valid_coco, path_valid_imgs)
    except AssertionError:
        print("\n Data set Already registered! Change string if want to re-register!")

    train_data = detectron2.data.datasets.load_coco_json(path_train_coco, path_train_imgs, dataset_name=train_name, extra_annotation_keys=None)
    valid_data = detectron2.data.datasets.load_coco_json(path_valid_coco, path_valid_imgs, dataset_name=valid_name, extra_annotation_keys=None)

    meta_train = MetadataCatalog.get(train_name).set(thing_classes=["pallet"])
    meta_valid = MetadataCatalog.get(valid_name).set(thing_classes=["pallet"])

    print("No of Train = ", len(train_data))
    print("No of Valid = ", len(valid_data))

    return train_data, meta_train, valid_data, meta_valid




"""
Visualisation for coco registered dataset.

"""


def visualize_data(dataset_name, no_of_display_imgs):
    """
    Visualize a random selection of images and annotations from a dataset.

    Args:
        dataset_name (str): Name of the dataset to visualize.
        no_of_display_imgs (int): Number of images to display.

    Returns:
        None
    """
    dataset_dicts = DatasetCatalog.get(dataset_name)
    c = 0

    for d in random.sample(dataset_dicts, no_of_display_imgs):
        img = cv2.imread(d['file_name'])
        visualizer = Visualizer(img[:, :, ::-1], metadata=meta_train, scale=0.5)
        out = visualizer.draw_dataset_dict(d)
        # c = c + 1

        image_rgb = cv2.cvtColor(out.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB)
        plt.imshow(image_rgb)
        plt.axis('off')  # Turn off axis labels
        plt.show()


""" 
Setting up the hyper parameters and training model.

"""

def start_training(PATH_TO_SAVE,train_name, valid_name,NUM_WORKERS,
                   IMS_PER_BATCH, BASE_LR, MAX_ITER, BATCH_SIZE_PER_IMAGE,
                   ):

    cfg = get_cfg()
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_regnety_4gf_dds_fpn_1x.yaml"))


    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

    cfg.DATASETS.TRAIN = (train_name,)
    cfg.DATASETS.TEST = (valid_name,)

    cfg.DATALOADER.NUM_WORKERS = NUM_WORKERS #4
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo

    cfg.SOLVER.IMS_PER_BATCH = IMS_PER_BATCH #8

    # cfg.SOLVER.momentum = 1.0
    cfg.SOLVER.BASE_LR = BASE_LR #0.0025 
    cfg.SOLVER.MAX_ITER = MAX_ITER #9000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset

    # epoch =  MAX_ITER * BATCH_SIZE / TOTAL_NUM_IMAGES


    cfg.TEST.EVAL_PERIOD = 500

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = BATCH_SIZE_PER_IMAGE #128  # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)

    cfg.MODEL.ANCHOR_GENERATOR.ANGLES = [[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 45,-45]]

    # # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
    cfg.OUTPUT_DIR = PATH_TO_SAVE
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = DefaultTrainer(cfg)

    trainer.resume_or_load(resume=True)
    print(cfg)
    # trainer.train()
    print("Finised Training! ")

def save_hyperparms(config,path_to_save_cvs):
    df = pd.DataFrame(list(config.items()), columns=['Hyperparameter', 'Value'])
    df.to_csv(path_to_save_cvs, index=False)
    print("Saved history at :",path_to_save_cvs,"\n\n")

def main():
    # Load configuration from YAML file
    torch.cuda.empty_cache()

    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    train_name = "pallet_train"
    valid_name = "pallet_valid"

    train_data, meta_train, valid_data, meta_valid = data_registration(train_name, valid_name,
                                                                        config['path_train_coco'],
                                                                        config['path_train_imgs'],
                                                                        config['path_valid_coco'],
                                                                        config['path_valid_imgs'])

    print("metadata: ", meta_train, meta_valid)

    # visualize_data function
    # dataset_name = train_name
    # number_of_images_to_display = 1
    # visualize_data(dataset_name, number_of_images_to_display)

    start_training(config['PATH_TO_SAVE'], train_name, valid_name, config['NUM_WORKERS'],
                config['IMS_PER_BATCH'], config['BASE_LR'], 
                config['MAX_ITER'], 
                config['BATCH_SIZE_PER_IMAGE']
                  )
    save_hyperparms(config,config['PATH_TO_SAVE_HISTORY'])
    

        

if __name__ == "__main__":
    main()