
## Machine Learning Scripts for training Pallet segmentation.  

This repo contians automated pipeline for training and analysis of model.


## Usage:
1. Make sure you have the dataset into below strucure:
- images:
  - 0001.JPG ....
- annotatons:
        
    - annotations_pallet.jsonn

       

2. Below command would start followig process:

    dataset spilting process --> training --> evaluation --> analysis

        ./start_training.sh


3. Goal is to do Pallet segmentation for measuring the length of pallet accurately in pixels.
   ![Alt text](https://github.com/prathamsss/pallet_Segmentation/blob/main/pallet-1.png)
      ![Alt text](https://github.com/prathamsss/pallet_Segmentation/blob/main/pallet-2.png)
