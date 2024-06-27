
## Machine Learning Scripts for training Pallet segmentation.  

This repo contians automated pipeline for training and analysis of model. Object segmentation model using MaskRCNN with Detectron2 is used as a framework. 



## Usage:
1. Make sure you have the dataset into below strucure:
- images:
  - 0001.JPG ....
- annotatons:
        
    - annotations_pallet.jsonn

       

2. Below command would start followig process:

    dataset spilting process --> training --> evaluation --> analysis

        ./start_training.sh


## Models (Old):

Use the following links for accessing the machine learning models and hyperparameters.\
Model hyperparameters:

        Hyperparameters: https://docs.google.com/spreadsheets/d/1GHMldCbbAou26cfFYkKpl1LtYK24sojHcdws-YC3w1k/edit?usp=sharing

Images:
    Train: 1801
    Val:  967
    Total: 2768

		Pallet Segmentation Model - 
		