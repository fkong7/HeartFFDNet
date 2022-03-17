# HeartFFDNet

This repository contains the source code for our paper:

Kong, F., Shadden, S.C., Whole Heart Mesh Generation For Image-Based Computational Simulations By Learning Free-From Deformations, MICCAI 2021

https://user-images.githubusercontent.com/31931939/125146359-4c264180-e0da-11eb-82b1-8bb1aacbfd2f.mp4

## Dependencies:

  - Tensorflow                     1.14
  - numpy                         1.18.4                          
  - scikit-learn                  0.22.2.post1           
  - scipy                         1.4.1                   
  - SimpleITK                     1.2.0rc2.dev1167+gd4cf2
  - vtk                           8.1.2                  

## Download Template files and Pre-trained Models
Please download the Code/examples folder from here:
https://drive.google.com/drive/folders/1tgoGWSyZwvafus4z8ueAxfrO_h2UwDE_?usp=sharing

## Create Control point grid and B-spline basis for training and testing
Below is the command to create trivariate B-spline tensor and relavant grid infomation for training and test. This only needs to be done once or alternatively, the .dat file is included in `examples/`
```
python template/make_control_grid.py \
  --template examples/template_with_veins_original.vtp \
  --template_im examples/place_holder_image.nii.gz \
  --num_pts 6  \
  --target_node_num 2048
```

## Prediction
Below is the command to use the pretrained model to generate whole heart mesh predictions on an example test image 
To generate simulation-ready mesh, please replace the mesh template with `template_simulation_left.vtp` for the left heart and `template_simulation_right.vtp` for the right heart.

```
python predict.py \
    --image examples \
    --mesh_dat examples/example_dat_of_template_with_veins.dat \
    --attr _test \
    --mesh_tmplt  examples/template_with_veins_original_normalized.vtp  \
    --model examples/weights_gcn.hdf5 \
    --output examples/output \
    --modality ct \
    --seg_id 1 2 3 4 5 6 7\
    --size 128 128 128 \
    --mode test \
    --amplify_factor 0.1 \
    --num_mesh 7 \
    --num_seg 1 \
    --num_block 3\
    --if_swap_mesh \
    --compare_seg
```
To reproduce the accuracy results of Multi-Res+WHS in Table 1, please use the following command. This will load the template without pulmonary veins or venae cavae and the model trained using the same dataset as used by Kong et. al. [8] that contains no pulmonary veins or venae cavae for a fair comparison with the previously reported methods. The MMWHS dataset can be obtained here: https://zmiclab.github.io/zxh/0/mmwhs/

```
python predict.py \
    --image /path/to/mmwhs/test/dataset \
    --mesh_dat examples/mmwhs_test/dat_of_template_with_no_veins.dat \
    --attr _test \
    --mesh_tmplt  examples/mmwhs_test/template_no_veins_normalized.vtp  \
    --model examples/mmwhs_test/weights_gcn_no_veins.hdf5 \
    --output examples/output/mmwhs_test \
    --modality ct \
    --seg_id 1 2 3 4 5 6 7\
    --size 128 128 128 \
    --mode test \
    --amplify_factor 0.1 \
    --num_mesh 7 \
    --num_seg 1 \
    --num_block 3\
    --if_swap_mesh \
    --compare_seg
```

## Training 
To train the model, please use the command below. We assume that the training data have already been pre-processed and written in .tfrecords format.
```
python train.py \
    --im_train /path/to/training/data/folder \
    --im_val /path/to/validation/data/folder \
    --attr 1 \
    --mesh /path/to/dat/file/containing/control/grid/info \
    --pre_train /path/to/any/pretrained/model/optional/weights_gcn.hdf5 \
    --output /path/to/output/folder \
    --modality ct \
    --num_epoch 100 \
    --batch_size 1 \
    --lr 0.001 \
    --size 128 128 128 \
    --cf_ratio 1. \
    --mesh_ids 0 1 2 3 4 5 6\
    --num_seg 1 \
    --num_block 3\
    --seg_weight 100 \
    --ctrl_weight 200 \
    --amplify_factor 0.1 \
    --shuffle_buffer_size 128
```
