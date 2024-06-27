""" 
Script to generate csv file having TP FP FN  for each image.
"""

from pycocotools.coco import COCO
import os
import pylab
import pandas as pd
import cv2
from pprint import pprint
import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.utils.coco as fouc
import json
import yaml
from pprint import pprint
import pandas as pd
import shutil



pylab.rcParams['figure.figsize'] = (10.0, 8.0)


def get_data_into_csv(GT_ANNO_FILE,DETECTION_FILE,
                      IMG_DIR, PATH_SAVE_CSV):
    print("==>")
    annType = ['segm','bbox']
    annType = annType[1]  # We select bbox
    print('Running demo for *%s* results.'%(annType))

    cocoGt = COCO(GT_ANNO_FILE)

    cocoDt=cocoGt.loadRes(DETECTION_FILE)
    print("total_images =", len(cocoDt.imgs))

    imgIds = sorted(cocoGt.getImgIds())

    data = []

    for imgId in range(len(cocoDt.imgs)-1):
        imgIds = cocoDt.getImgIds(imgIds =imgId)
        img = cocoDt.loadImgs(imgIds)
    
        file_name = img[0]['file_name']
        img = cv2.imread(os.path.join(IMG_DIR,file_name))
        height, width, _ = img.shape

        annIds = cocoDt.getAnnIds(imgIds=imgId ,iscrowd=None)

        anns = cocoDt.loadAnns(annIds)
        area_of_img =  height * width
        
        if len(anns) == 0:
            data.append({'file_name': file_name, 
                    'predicted area': None,
                    'image height': height, 
                    'image width': width,
                    '%_of_predicted_area':None
                    })
        else:
            
            pred = (anns[0])

            ratio = (pred['area'] / area_of_img)*100
            
            
            data.append({'file_name': file_name, 
                        'predicted area': pred['area'],
                        'image height': height, 
                        'image width': width,
                        '%_of_predicted_area':ratio

                        })

        
    df = pd.DataFrame(data)




    df.to_csv(PATH_SAVE_CSV, index=False)  # Replace 'results.csv' with your desired filename
    return df


def load_fiftyone(DETECTION_FILE,IMG_DIR,GT_ANNO_FILE,
                  CSV_FILE_PATH):
    
    dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        data_path= IMG_DIR,
        labels_path= GT_ANNO_FILE,
        include_id=True,
    )

    
    dataset.default_classes = ["pallet"]

    with open(DETECTION_FILE, 'r') as f:
        final_coco_data = json.load(f)
        print("total evaluation on ==>",len(final_coco_data))
    
    # Add COCO predictions to `predictions` field of dataset
        
    classes = dataset.default_classes

    fouc.add_coco_labels(dataset, "predictions", 
                        final_coco_data, classes,
                        coco_id_field='coco_id',
                        include_annotation_id=True)
    
    dataset.rename_sample_field("segmentations", "ground_truth")

    results = dataset.evaluate_detections(
                                        "predictions",
                                        gt_field="ground_truth",
                                        method="coco",
                                        eval_key="eval",
                                        iou=0.88
                                        )
    
    print("TP: %d" % dataset.sum("eval_tp"))
    print("FP: %d" % dataset.sum("eval_fp"))
    print("FN: %d" % dataset.sum("eval_fn"))
    print('Report :',results.print_report())


    # Load the existing CSV file into a DataFrame
    df = pd.read_csv(CSV_FILE_PATH)
    # Iterate over samples
    for sample in (dataset):
        filename = (sample.filepath).split('/')[-1]
        
        # pprint(sample.eval)
        
        if filename in df['file_name'].values:

            idx = df.index[df['file_name'] == filename][0]
            print("idx ==>",idx,df.index[df['file_name'] == filename])
        
            df.loc[idx, 'fp'] = int(sample.eval_fp)
            df.loc[idx, 'tp'] = int(sample.eval_tp)
            df.loc[idx, 'fn'] = int(sample.eval_fn)
            
            

        df.to_csv(CSV_FILE_PATH, 
                  index=False)
        print("Saved CSV File at ",CSV_FILE_PATH)



if __name__ == "__main__":
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    DETECTION_FILE = os.path.join(config['PATH_TO_SAVE_EVALUATION'],"coco_instances_results.json")

    get_data_into_csv(config['COCO_FILE_TO_EVAL'],DETECTION_FILE,
                      config['IMAGES_PATH'], config['PATH_SAVE_CSV']
                      )
    
    load_fiftyone(DETECTION_FILE,config['IMAGES_PATH'],
                  config['COCO_FILE_TO_EVAL'],config['PATH_SAVE_CSV']
                  )
    
    
    
    