import cv2
import random
import os
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog


class SegPredictor:
    """ 
    A class to perform pallet segmentation on images.
    """

    def __init__(self, model_weights_path):
        """ 
        Initializes the SegPredictor object.

        Args:
        - model_weights_path (str): Path to the model weights.
        """
        cfg = get_cfg()
        cfg.merge_from_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        cfg.MODEL.WEIGHTS = model_weights_path
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.50
        cfg.TEST.DETECTIONS_PER_IMAGE = 2
        cfg.MODEL.RPN.PRE_NMS_TOPK_TEST=2


        self.predictor = DefaultPredictor(cfg)

    def predict(self, img_path, save_path):
        """ 
        Predicts pallet segmentation on an image.

        Args:
        - img_path (str): Path to the input image.
        - save_path (str): Path to save the output image.
        """
        image = cv2.imread(img_path)
        model_out = self.predictor(image)
        out_mask = model_out["instances"].pred_masks.to("cpu").numpy()
        score = model_out["instances"].scores.to("cpu").numpy()
        out_mask = out_mask.squeeze()
        binary_mask = out_mask.astype('uint8') * 255
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_image = cv2.imread(img_path)
        cv2.drawContours(contour_image, contours, -1, (0, 0, 255), 17)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y = largest_contour[0][0]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        font_color = (0, 0, 255)
        font_thickness = 6
        cv2.putText(contour_image, str(score[0]), (x, y - 50), font, font_scale, font_color, font_thickness)
        cv2.imwrite(save_path, contour_image)


def register_datasets(valid_name, path_valid_coco, path_valid_imgs):
    """
    Registers COCO instances for training and validation datasets and loads the data.

    Args:
    - valid_name (str): Name for the validation dataset.
    - path_valid_coco (str): Path to the validation COCO annotation file.
    - path_valid_imgs (str): Path to the folder containing validation images.

    Returns:
    - valid_data (list): Validation data.
    - meta_valid (MetadataCatalog): Metadata for the validation dataset.
    """
    # Register datasets
    # register_coco_instances(train_name, {}, path_train_coco, path_train_imgs)
    register_coco_instances(valid_name, {}, path_valid_coco, path_valid_imgs)

    # Load data
    # train_data = DatasetCatalog.get(train_name)
    valid_data = DatasetCatalog.get(valid_name)
    # meta_train = MetadataCatalog.get(train_name)
    meta_valid = MetadataCatalog.get(valid_name)

    return valid_data, meta_valid


def visualize_predicted_images(dataset_dicts, predictor, output_folder):
    """
    Visualizes predicted images and saves them.

    Args:
    - dataset_dicts (list): List of dataset dictionaries.
    - predictor (SegPredictor): SegPredictor object for prediction.
    - output_folder (str): Path to the folder to save the visualizations.
    """
    for d in dataset_dicts:
        img_path = d["file_name"]
        img_to_save = os.path.split(img_path)[-1]
        path_to_save = os.path.join(output_folder, img_to_save)
        try:
            predictor.predict(img_path, path_to_save)
        except Exception as e:
            print(f"Exception for {img_path}: {e}")


def main():
    train_name = "pallet_train1"
    valid_name = "pallet_valid1"

    
    path_valid_coco = "/home/ir/Desktop/ai_model_development/PalletSegmentation/annotation/test_modified_coco_annotation.json"
    path_valid_imgs = "/home/ir/Desktop/ai_model_development/PalletSegmentation/final_dataset/images"

    valid_data, meta_valid = register_datasets(valid_name, path_valid_coco, path_valid_imgs)

    seg = SegPredictor("/home/ir/Desktop/ai_model_development/PalletSegmentation/output/img_2906_e_40_lr_0.0001_batch_8.pth")

    output_folder = "/home/ir/Desktop/temp_valid_inf"
    dataset_dicts = random.sample(valid_data, 561)
    visualize_predicted_images(dataset_dicts, seg, output_folder)


if __name__ == "__main__":
    main()
