# DLCV-Project
                                                                                                                                                                                                                                                                                      DLCV Course Project
                                                                                                                                                                                                                             Face Detection under Adverse Illumination Conditions (Low Light Face Detection)

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
We have developed following face detection models
1. DSFD 
2.DSFD+DCENet 
3.DSFD+DCENet+Deblur GAN
5.DSFD+DCENet+SRGAN 
6.DSFD+DCENet+MOCO2

Enviorment requirement
1. Pytorch
2. Numpy
3. Pandas
4. opencv
5. Easydict

 DSFD+DCENet+MOCO2
 To train the the models weight
- Pretrain the model:
   store the wider face in "/content/drive/MyDrive/low-light-face-detection/train_code/dataset/WiderFace/WIDER_train/images" and noised wider face data in "/content/drive/MyDrive/low-light-face-detection/train_code/dataset/WiderFace/WIDER_images_add_noise" folder
 
-To run the pretrained code
first change the directory 
!python3 pretrain_fast_headers.py
 above python code uses self supervised learning to pretrain on wider face data ,.to use it for downsampling purpose, we need to remove the header from the base encoder , to do that we need to run the transfer.py file

!python3 transfer_headers.py

It will save the pretrained weights to pretrain folder.
-To run the train.py
Place the Dark face dataset in "train_code/dataset/DarkFace/"
then run

!python3 train.py

above code saves checkpoint for every 2500 iterations
 save the trained weights in  "/content/drive/MyDrive/DLCV Project/FaceDetection-DSFD-MOCO2/weights" folder.

-To run the test.py 
First change the directory
run this code - !pip3 install torch==1.2.0+cu92 torchvision==0.4.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html
As you may run into pytorch version issue
then run  !python3 test.py

test.py takes dark images as input and provides coordinates of bounding box and confience of each faces detected as output.

to run the demo.py , you need to mention the visual threshold value, the folder name and the test file path relative to the folder. 

!python3 demo.py --visual_threshold 0.5 --model 'FaceDetection-DSFD-MOCO2' --test_path 'test2.png'

We have stored 10 test images and their corresponding text files for annonations in Dataset folder. if you want to test any of our models ,you need to place those images and their tags in the directory of that model. 
