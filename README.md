# AI-based-microscopy-system-project
This folder contains codes for counting and detecting blood cells (RBC, WBC, and platelets) from dataset images and in real-time, for the detection of malaria, and the early detection of leukemia (blood cancer).
The first code is for blood cells counter. It is coded inspired with the CBC test (complete blood count) that is performed in laboratory, in order to remove human error and time consumption. Two GitHub and Kaggle sources were used to create the CBC dataset through the combination
of BCCD images totaling 364 images that needed YOLO conversion and TXL-PBC images totaling 1,440 labeled in YOLO format. The CBC images increased to 12,620 when data augmentation was applied ten times. The dataset can be found here as well. 
This model is trained on YOLOv8, a specialized computer vision pre-trained model, trained specially on images for the purpose of object detection.
The model achieved metrics of 82%. 
The second model is for malaria detection which is trained on vgg19, on a dataset collected from NIH containing total 27,000 images with two categories: parasitized and uninfected. The model achieved 89% accuracy. 
The third model is for the early detection of leukemia, trained on EfficientNetB0, on a dataset combining normal blood cell images with abnormal blood cell images that amount to 17,101 normal cells and 6,510 leukemic cells. This model achieved 98% accuracy. 
The last code is for the GUI. 
