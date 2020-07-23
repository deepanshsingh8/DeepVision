* For task 1, to train the model and create masks for each dataset, run the following command line (after cd into folder 'code'):
python main.py --dataset_name <name_of_dataset> --epochs <number_of_epochs>

* For example, to train the model and create masks for DIC-C2DH-HeLa using 100 epochs, run the following command line:
python main.py --dataset_name DIC-C2DH-HeLa --epochs 100

NOTE: 
* to make sure the model runs correctly first, try number of epochs = 1. 
* --epochs is optional (if omitted, 100 epochs will be used as default).
* to train using GPU, use: 
python --dataset_name DIC-C2DH-HeLa --device cuda:0