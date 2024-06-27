#!/usr/bin/encv python
import torch, detectron2
# get_ipython().system('nvcc --version')
# TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
# CUDA_VERSION = torch.__version__.split("+")[-1]
# print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
# print("detectron2:", detectron2.__version__)


# Some basic setup:
import detectron2
import os, random
import cv2, json
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.data import detection_utils as utils
import detectron2.data
import matplotlib.pyplot as plt
import yaml
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
import os
import detectron2.data
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader



def data_registration(data_name,coco_file_path,images_path):
    """ 
    Data registration for detectron2 evaluation
    """
    images_path = "/home/ai-infinium/Desktop/Dataset/all_data/images"
   
    print("Total Train Images in folder => ", len(os.listdir(images_path)))
    
    try:
        register_coco_instances(data_name, {}, coco_file_path, images_path)
    except AssertionError:
        print("\n Data set Already registered! Change string if want to re-register!")

    dataset = detectron2.data.datasets.load_coco_json(coco_file_path, images_path, dataset_name=data_name, extra_annotation_keys=None)

    meta_data = MetadataCatalog.get(data_name).set(thing_classes=["pallet"])

    print("No of Train = ", len(dataset))

    return dataset, meta_data

def evaluation(dataset, PATH_TO_SAVE,MODEL_WEIGHTS_FILE):
    """ 
      Setting up the hyper parameters and training model.
    """
     
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = (dataset,)

    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    cfg.MODEL.ANCHOR_GENERATOR.ANGLES = [[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 45,-45]]
    
    cfg.OUTPUT_DIR = PATH_TO_SAVE
    cfg.MODEL.WEIGHTS = MODEL_WEIGHTS_FILE  # path to the model we just trained

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.90
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.0
    cfg.TEST.DETECTIONS_PER_IMAGE = 1
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST=2
    predictor = DefaultPredictor(cfg)

    # Get COCO Evaluation on validation set

    evaluator = COCOEvaluator(data_name, 
                            output_dir = PATH_TO_SAVE)
    loader = build_detection_test_loader(cfg, data_name)
    print(inference_on_dataset(predictor.model, loader, evaluator))

    return predictor

     

def modify_detection_file(DETECTION_FILE):
    with open(DETECTION_FILE, 'r') as file:
        data = json.load(file)
    

    # Iterate through each item and modify the category_id
    for item in data:
        if item['category_id'] == 0:
            item['category_id'] = 1

    # Write the modified JSON data back to the file
    with open(DETECTION_FILE, 'w') as file:
        json.dump(data, file, indent=2)



    print("Category IDs updated successfully.")




if __name__ == "__main__":

    with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)

    data_name = "pallet_train"

    dataset, metadata = data_registration(data_name,
                                        config['COCO_FILE_TO_EVAL'],config['IMAGES_PATH'],
                                        )

    MODEL_WEIGHTS_FILE =  os.path.join(config['PATH_TO_SAVE_EVALUATION'],
                                    'model_final.pth')

    evaluation(dataset, 
            config['PATH_TO_SAVE_EVALUATION'],
            MODEL_WEIGHTS_FILE)
    
    DETECTION_FILE = os.path.join( config['PATH_TO_SAVE_EVALUATION'],
                                  "coco_instances_results.json")
    
    modify_detection_file(DETECTION_FILE)