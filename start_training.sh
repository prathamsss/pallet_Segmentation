#!/bin/bash

# You should have a directory having all images and directory having all annotation file.

# First we process with spilting annotation into train and test 

PATH_TO_COCO_SPLIT_DIR="/home/ai-infinium/Desktop/ai_model_development/cocosplit"
PATH_TO_ALL_ANNOTATION_FILE="/home/ai-infinium/Desktop/Dataset/all_data/annotations/merged_pallet.json"
PATH_TO_SAVE_TRAIN_TEST_SPILT_FILE="/home/ai-infinium/Desktop/Dataset/all_data/annotations"
TRAIN_TEST_SPILT=0.75
python3 "$PATH_TO_COCO_SPLIT_DIR/cocosplit.py" -s $TRAIN_TEST_SPILT "$PATH_TO_ALL_ANNOTATION_FILE"  "$PATH_TO_SAVE_TRAIN_TEST_SPILT_FILE/train.json" "$PATH_TO_SAVE_TRAIN_TEST_SPILT_FILE/test.json"

# Here We start the training 

# make sure you have configured yaml file in same directory.
python3 training_pipeline.py

# Evaluation of model this would generate json file containing detections

python3 evaluation.py

python3 model_analysis.py