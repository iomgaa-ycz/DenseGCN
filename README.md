
This code repository is a test repository for the paper "DenseGCN: A Multi-level and Multi-temporal Graph Convolutional Network for Action Recognition", used to test DenseGCN on the NTU RGB+D dataset and the NTU RGB+D 120 dataset.

## Download Data

- Download the raw data of [NTU-RGBD](https://github.com/shahroudy/NTURGB-D) and [NTU-RGBD120](https://github.com/shahroudy/NTURGB-D). Put NTU-RGBD data under the directory `./data/nturgbd_raw`. Put NTU-RGBD120 data under the directory `./data/nturgbd120_raw`. 
 

## Data Preparation

- For NTU-RGBD, preprocess data with python data_gen/ntu_gendata.py. 
- For NTU-RGBD120, preprocess data with python data_gen/ntu120_gendata.py.

## Update cuda extension

  ```
  cd ./model/Temporal_shift
  bash run.sh
  ```
## model
The model weights file is saved in `. /weights`.

## Test
If you wish to run the test program, run the following command.
- Testing the accuracy of DenseGCN in an x-view benchmark of the NTU RGB+D dataset.
  ```
  python test.py --mode ntu_xview
  ```
  
- Testing the accuracy of DenseGCN in an X-Sub benchmark of the NTU RGB+D dataset.
  ```
  python test.py --mode ntu_sub
  ```
  
- Testing the accuracy of DenseGCN in an X-Set120 benchmark of the NTU RGB+D 120 dataset.
  ```
  python test.py --mode ntu_xset120
  ```
  
- Testing the accuracy of DenseGCN in an X-Sub benchmark of the NTU RGB+D 120 dataset.
  ```
  python test.py --mode ntu_sub120
  ```