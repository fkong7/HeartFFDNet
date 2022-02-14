#Copyright (C) 2021 Fanwei Kong, Shawn C. Shadden, University of California, Berkeley

#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
sys.path.append(os.path.join(os.path.dirname(__file__), "external"))
import glob
import functools
import pickle
import argparse
import numpy as np
from sklearn.model_selection import train_test_split

import tensorflow as tf
print("TENSORFLOW VERSION: ", tf.__version__)
from tensorflow.python.keras.optimizers import Adam, SGD
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import losses
from tensorflow.python.keras import models

from utils import getTrainNLabelNames,get_model_memory_usage, buildImageDataset, construct_feed_dict 
from custom_layers import *

from augmentation import changeIntensity_img, _augment
from dataset import get_baseline_dataset
from model import HeartDeepFFD
from loss import *
from call_backs import *
from vtk_utils.vtk_utils import *

import SimpleITK as sitk
"""# Set up"""

parser = argparse.ArgumentParser()
parser.add_argument('--im_train',  help='Name of the folder containing the image data')
parser.add_argument('--im_trains', nargs='+', help='Name of the folder containing the image data')
parser.add_argument('--im_vals', nargs='+', help='Name of the folder containing the image data')
parser.add_argument('--file_pattern', default='*.tfrecords', help='Pattern of the .tfrecords files')
parser.add_argument('--pre_train_im', default='', help="Filename of the pretrained unet")
parser.add_argument('--pre_train', default='', help="Filename of the pretrained model")
parser.add_argument('--attr_trains', nargs='+', help='Attribute name of the folders containing tf records')
parser.add_argument('--attr_vals', nargs='+', help='Attribute name of the folders containing tf records')
parser.add_argument('--train_data_weights', type=float, nargs='+', help='Weights to apply for the samples in different datasets')
parser.add_argument('--val_data_weights', type=float, nargs='+', help='Weights to apply for the samples in different datasets')
parser.add_argument('--pre_train_num_epoch', type=int, default=300, help="Number of epochs for training with geometric mean loss")
parser.add_argument('--mesh',  help='Name of the .dat file containing mesh info')
parser.add_argument('--output',  help='Name of the output folder')
parser.add_argument('--modality', nargs='+', help='Name of the modality, mr, ct, split by space')
parser.add_argument('--num_epoch', type=int, help='Maximum number of epochs to run')
parser.add_argument('--num_seg', type=int,default=8, help='Number of segmentation classes')
parser.add_argument('--num_block', type=int,default=3, help='Number of graph conv block')
parser.add_argument('--disable_block_loss', type=int, nargs='+', default=None, help='Turn of supervision for blocks')
parser.add_argument('--seg_weight', type=float, default=1., help='Weight of the segmentation loss')
parser.add_argument('--ctrl_weight', type=float, default=1., help='Weight of the ctrl pts reg loss')
parser.add_argument('--im_weight', type=float, default=1., help='Weight of the image reg loss')
parser.add_argument('--geom_weights', type=float, default=[0.5, 0.5], nargs='+', help='Weight of the ctrl pts reg loss')
parser.add_argument('--mesh_ids', nargs='+', type=int, default=[2], help='Number of meshes to train')
parser.add_argument('--seed', type=int, default=41, help='Randome seed')
parser.add_argument('--batch_size', type=int, default=10, help='Batch size')
parser.add_argument('--shuffle_buffer_size', type=int, default=128, help='Shuffle buffer size')
parser.add_argument('--lr', type=float, help='Learning rate')
parser.add_argument('--cf_ratio', type=float, default=1., help='Loss ratio between gt chamfer loss and pred chamfer loss')
parser.add_argument('--size', type = int, nargs='+', help='Image dimensions')
parser.add_argument('--hidden_dim', type = int, default=128, help='Hidden dimension')
parser.add_argument('--amplify_factor', type=float, default=1., help="amplify_factor of the predicted displacements")
parser.add_argument('--if_warp_im', action='store_true', help='If to deform image too')
args = parser.parse_args()
print('Finished parsing...')

modality = args.modality
seed = args.seed
epochs = args.num_epoch
batch_size = args.batch_size
img_shape = args.size
img_shape = (img_shape[0], img_shape[1], img_shape[2], 1)
lr = args.lr

save_loss_path = args.output
save_model_path = os.path.join(args.output, "weights_gcn.hdf5")


""" Create new directories """
try:
    os.makedirs(os.path.dirname(save_model_path))
    os.makedirs(os.path.dirname(save_loss_path))
except Exception as e: print(e)


"""# Feed in mesh info"""

pkl = pickle.load(open(args.mesh, 'rb'))
mesh_info = construct_feed_dict(pkl)

"""# Build the model"""
model = HeartDeepFFD(batch_size, img_shape, args.hidden_dim, mesh_info, amplify_factor=args.amplify_factor,num_mesh=len(args.mesh_ids), num_seg=args.num_seg, num_block=args.num_block,if_warp_im=args.if_warp_im)

"""# Build dataset iterator"""
unet_gcn = model.build_keras('bspline')
#unet_gcn = model.build_conv_ffd()
if args.pre_train_im != '':
    unet_gcn = model.load_pre_trained_weights(unet_gcn, args.pre_train_im,trainable=False)
unet_gcn.summary(line_length=150)
adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6, amsgrad=True)
output_keys = [node.op.name.split('/')[0] for node in unet_gcn.outputs]
print("Output Keys: ", output_keys)
ctrl_loss_list = []
im_loss_list = []
grid_weight = K.variable(args.ctrl_weight)
for i in range(args.num_block):
    ctrl_loss_list.append(ctrl_pts_loss(grid_weight))
    if args.if_warp_im:
        im_loss_list.append(image_warp_loss(args.im_weight, mesh_info['image_data']))
if args.num_seg >0:
    #losses = [ mesh_loss_geometric_cf(mesh_info, 3, sub_loss_weights, args.cf_ratio, mesh_info['edge_length_scaled'][(i-1)%len(args.mesh_ids)]) for i in range(1, len(output_keys))]
    #losses = [ mesh_loss_geometric_cf(mesh_info, args.geom_weights, i % len(args.mesh_ids), args.cf_ratio) for i in range(len(output_keys)-1-args.num_block)]
    losses = [ mesh_point_loss_cf(args.cf_ratio)  for i in range(len(output_keys)-1-args.num_block)]
    #losses = [bce_dice_loss] + losses
    losses = [binary_bce_dice_loss] + im_loss_list + ctrl_loss_list + losses 
else:
    #losses = [ mesh_loss_geometric_cf(mesh_info, 3, sub_loss_weights, args.cf_ratio, mesh_info['edge_length_scaled'][i%len(args.mesh_ids)]) for i in range(len(output_keys))]
    #losses = ctrl_loss_list + [ mesh_loss_geometric_cf(mesh_info, args.geom_weights, i % len(args.mesh_ids), args.cf_ratio) for i in range(len(output_keys)-1-args.num_block)]
    losses = im_loss_list + ctrl_loss_list + [ mesh_point_loss_cf(args.cf_ratio) for i in range(len(output_keys)-1-args.num_block)]

losses = dict(zip(output_keys, losses))

metric_loss = []
metric_key = []
for i in range(1, len(args.mesh_ids)+1):
    metric_key.append(output_keys[-i])
    metric_loss.append(mesh_point_loss_cf(args.cf_ratio))
print(metric_key)
metrics_losses = dict(zip(metric_key, metric_loss))
metric_loss_weights = list(np.ones(len(args.mesh_ids)))
loss_weights = list(np.ones(len(output_keys)))
# turn of losses for the first few deformation blocks
if args.disable_block_loss is not None:
    for i in args.disable_block_loss:
        if i >= 0 and i<args.num_block:
            im_block = args.num_block if args.if_warp_im else 0
            grid_block = args.num_block
            loss_weights[1+i] = 0. # for im_warp loss
            loss_weights[1+im_block+i] = 0. # for control grid loss
            loss_weights[(1+grid_block+im_block+i*len(args.mesh_ids)):(1+grid_block+im_block+(i+1)*len(args.mesh_ids))] = [0.]*len(args.mesh_ids)
        else:
            print("Invalid block ID, loss not turned off")
if args.num_seg > 0:
    loss_weights[0] = args.seg_weight
# Encourage larger movements on block 2, 3 by setting the loss to be negative
#loss_weights[2] *= -1.
#loss_weights[3] *= -1.
print("Current loss weights: ", loss_weights)


unet_gcn.compile(optimizer=adam, loss=losses,loss_weights=loss_weights,  metrics=metrics_losses)
""" Setup model checkpoint """
save_model_path = os.path.join(args.output, "weights_gcn.hdf5")
save_model_path2 = os.path.join(args.output,  "weights_gcn-{epoch:02d}.hdf5")

cp_cd = SaveModelOnCD(metric_key, save_model_path, patience=50)
#cp = tf.keras.callbacks.ModelCheckpoint(filepath=save_model_path, save_weights_only=True, period=100)
#cp = tf.keras.callbacks.ModelCheckpoint(filepath=save_model_path, monitor='val_loss', save_best_only=True, verbose=1)
#cp_time_lap = tf.keras.callbacks.ModelCheckpoint(filepath=save_model_path2, verbose=1, save_weights_only=True,period=2)
lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.000005)
weight_schedule = ReduceLossWeight(grid_weight, patience=5, factor=0.95)
#erly_stp = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)
call_backs = [cp_cd,lr_schedule, weight_schedule]
# Alternatively, load the weights directly: model.load_weights(save_model_path)

try:
    #unet_gcn = models.load_model(save_model_path, custom_objects=custom_objects)
    if args.pre_train != '':
        print("Loading pre-trained model: ", args.pre_train)
        unet_gcn.load_weights(args.pre_train)
    else:
        unet_gcn.load_weights(save_model_path)
except Exception as e:
  print("Model not loaded", e)
  pass

"""## Set up train and validation datasets
Note that we apply image augmentation to our training dataset but not our validation dataset.
"""
tr_cfg = {
    'changeIntensity': {"scale": [0.9, 1.1],"shift": [-0.1, 0.1]}, 
}
tr_preprocessing_fn = functools.partial(_augment, **tr_cfg)

val_cfg = {}

if_seg = True if args.num_seg>0 else False
val_preprocessing_fn = functools.partial(_augment, **val_cfg)

train_ds_list, val_ds_list = [], []
train_ds_num, val_ds_num = [], []
for data_folder_out, attr in zip(args.im_trains, args.attr_trains):
    x_train_filenames_i = buildImageDataset(data_folder_out, args.modality, args.seed, mode='_train'+attr, ext=args.file_pattern)
    #x_train_filenames_i = [buildImageDataset(data_folder_out, args.modality, args.seed, mode='_val'+attr, ext=args.file_pattern)[0]]
    #print("train data debug: ", x_train_filenames_i)
    train_ds_num.append(len(x_train_filenames_i))
    train_ds_i = get_baseline_dataset(x_train_filenames_i, preproc_fn=tr_preprocessing_fn, mesh_ids=args.mesh_ids, \
            shuffle_buffer=args.shuffle_buffer_size, if_seg=if_seg, num_block=args.num_block,if_warp_im=args.if_warp_im)
    train_ds_list.append(train_ds_i)
for data_val_folder_out, attr in zip(args.im_vals, args.attr_vals):
    x_val_filenames_i = buildImageDataset(data_val_folder_out, args.modality, args.seed, mode='_val'+attr, ext=args.file_pattern)
    #x_val_filenames_i = [buildImageDataset(data_val_folder_out, args.modality, args.seed, mode='_val'+attr, ext=args.file_pattern)[0]]
    #print("val data debug: ", x_val_filenames_i)
    val_ds_num.append(len(x_val_filenames_i))
    val_ds_i = get_baseline_dataset(x_val_filenames_i, preproc_fn=val_preprocessing_fn, mesh_ids=args.mesh_ids, \
            shuffle_buffer=args.shuffle_buffer_size, if_seg=if_seg, num_block=args.num_block, if_warp_im=args.if_warp_im)
    val_ds_list.append(val_ds_i)
train_data_weights = [w/np.sum(args.train_data_weights) for w in args.train_data_weights]
val_data_weights = [w/np.sum(args.val_data_weights) for w in args.val_data_weights]
print("Sampling probability for train and val datasets: ", train_data_weights, val_data_weights)
train_ds = tf.data.experimental.sample_from_datasets(train_ds_list, weights=train_data_weights)
train_ds = train_ds.batch(args.batch_size)
val_ds = tf.data.experimental.sample_from_datasets(val_ds_list, weights=val_data_weights)
val_ds = val_ds.batch(args.batch_size)

num_train_examples = train_ds_num[np.argmax(train_data_weights)]/np.max(train_data_weights)
#num_train_examples = 300 # set the same for comparison
num_train_examples = 1500
num_val_examples =  val_ds_num[np.argmax(val_data_weights)]/np.max(val_data_weights) 
print("Number of train, val samples after reweighting: ", num_train_examples, num_val_examples)
""" Print Layer Name """
#layer_id = list()
#for i, layer in enumerate(unet_gcn.layers):
#    print(i, layer.name)
#    if 'projection' in layer.name:
#        layer_id.append(i)
#print("Projection layer id: ", layer_id)
""" Training """
history =unet_gcn.fit(train_ds, 
                   steps_per_epoch=int(np.ceil(num_train_examples / float(batch_size))),
                   epochs=args.pre_train_num_epoch,
                   validation_data=val_ds,
                   validation_steps= int(np.ceil(num_val_examples / float(batch_size))),
                   callbacks=call_backs)
with open(save_loss_path+"_history", 'wb') as handle: # saving the history 
        pickle.dump(history.history, handle)
#
#data_aug_iter = train_ds.make_one_shot_iterator()
#next_element = data_aug_iter.get_next()
#inputs, outputs = None, None
#with tf.Session() as sess: 
#    inputs, outputs = sess.run(next_element)
##losses = unet_gcn.evaluate(inputs, outputs)
#preds = unet_gcn.predict(inputs, batch_size=1)
#write_numpy_points(np.squeeze(outputs[-1]), 'gt_last.vtp')
#write_numpy_points(np.squeeze(preds[-1]), 'pred_lass.vtp')
##print(losses)
##print(unet_gcn.metrics_names)
