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
import numpy as np
import glob
import re
try:
    import tensorflow as tf
    from tensorflow.python.keras import backend as K
except Exception as e: print(e)

def dice_score(pred, true):
    pred = pred.astype(np.int)
    true = true.astype(np.int)
    num_class = np.unique(true)

    #change to one hot
    dice_out = [None]*len(num_class)
    for i in range(1, len(num_class)):
        pred_c = pred == num_class[i]
        true_c = true == num_class[i]
        dice_out[i] = np.sum(pred_c*true_c)*2.0 / (np.sum(pred_c) + np.sum(true_c))

    mask =( pred > 0 )+ (true > 0)
    dice_out[0] = np.sum((pred==true)[mask]) * 2. / (np.sum(pred>0) + np.sum(true>0))
    return dice_out

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

def buildImageDataset(data_folder_out, modality, seed, mode='_train', ext='*.tfrecords'):
    import random
    x_train_filenames = []
    filenames = [None]*len(modality)
    nums = np.zeros(len(modality))
    for i, m in enumerate(modality):
      filenames[i], _ = getTrainNLabelNames(data_folder_out, m, ext=ext, fn=mode)
      nums[i] = len(filenames[i])
      x_train_filenames+=filenames[i]
      #shuffle
      random.shuffle(x_train_filenames)
    random.shuffle(x_train_filenames)      
    print("Number of images obtained for training and validation: " + str(nums))
    return x_train_filenames

def construct_feed_dict(pkl):
    """Construct feed dictionary."""
    feed_dict = dict()
    try:
        feed_dict['image_data'] = pkl['image_data']
        feed_dict['ffd_matrix_image'] = [tf.SparseTensor(indices=i[0], values=i[1].astype(np.float32), dense_shape=i[-1]) for i in pkl['ffd_matrix_image']]
    except Exception as e:
        print(e)
        feed_dict['image_data'] = None
        feed_dict['ffd_matrix_image'] = None
    #feed_dict['tmplt_coords'] = pkl['tmplt_coords'].astype(np.float32)
    feed_dict['grid_coords'] = tf.convert_to_tensor(pkl['grid_coords'], dtype=tf.float32)
    feed_dict['sample_coords'] = tf.convert_to_tensor(pkl['sample_coords'], dtype=tf.float32)
    feed_dict['ffd_matrix_mesh'] = [tf.SparseTensor(indices=i[0], values=i[1].astype(np.float32), dense_shape=i[-1]) for i in pkl['ffd_matrix_mesh']]
    feed_dict['grid_downsample'] = [tf.SparseTensor(indices=i[0], values=i[1].astype(np.float32), dense_shape=i[-1]) for i in pkl['grid_downsample']]
    feed_dict['grid_upsample'] = [tf.SparseTensor(indices=i[0], values=i[1].astype(np.float32), dense_shape=i[-1]) for i in pkl['grid_upsample']]
    feed_dict['struct_node_ids'] = pkl['struct_node_ids']
    feed_dict['tmplt_faces'] = [tf.convert_to_tensor(faces, dtype=tf.int32) for faces in pkl['tmplt_faces']]
    #feed_dict['adjs']= [tf.SparseTensor(indices=j[0], values=j[1].astype(np.float32), dense_shape=j[-1]) for j in pkl['support']]
    feed_dict['adjs']= [[tf.SparseTensor(indices=j[0], values=j[1].astype(np.float32), dense_shape=j[-1]) for j in l] for l in pkl['support']]
    return feed_dict

def get_model_memory_usage(batch_size, model):
    import numpy as np
    from tensorflow.python.keras import backend as K

    shapes_mem_count = 0
    internal_model_mem_count = 0
    for l in model.layers:
        layer_type = l.__class__.__name__
        if layer_type == 'Model':
            internal_model_mem_count += get_model_memory_usage(batch_size, l)
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

    number_size = 4.0
    if K.floatx() == 'float16':
         number_size = 2.0
    if K.floatx() == 'float64':
         number_size = 8.0

    total_memory = number_size*(batch_size*shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3) + internal_model_mem_count
    return gbytes

def getTrainNLabelNames(data_folder, m, ext='*.nii.gz',fn='_train', seg_fn='_masks'):
  x_train_filenames = []
  y_train_filenames = []
  print("DEBUG: ", os.path.join(data_folder,m+fn,ext))
  for subject_dir in sorted(glob.glob(os.path.join(data_folder,m+fn,ext))):
      x_train_filenames.append(os.path.realpath(subject_dir))
  try:
      for subject_dir in sorted(glob.glob(os.path.join(data_folder ,m+fn+seg_fn,ext))):
          y_train_filenames.append(os.path.realpath(subject_dir))
  except Exception as e: print(e)

  return x_train_filenames, y_train_filenames

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def np_to_tfrecords(X, Y, file_path_prefix=None, verbose=True, debug=True):
            
    if Y is not None:
        assert X.shape == Y.shape

    # Generate tfrecord writer
    result_tf_file = file_path_prefix + '.tfrecords'
    writer = tf.python_io.TFRecordWriter(result_tf_file)
    if verbose:
        print("Serializing example into {}".format(result_tf_file))
        
    # iterate over each sample,
    # and serialize it as ProtoBuf.
    
    d_feature = {}
    d_feature['X'] = _float_feature(X.flatten())

    if debug:
        print("**** X ****")
        print(X.shape, X.flatten().shape)
        print(X.dtype)
    if Y is not None:
        d_feature['Y'] = _int64_feature(Y.flatten())
        if debug:
            print("**** Y shape ****")
            print(Y.shape, Y.flatten().shape)
            print(Y.dtype)

    #first axis is the channel dimension
    d_feature['shape0'] = _int64_feature([X.shape[0]])
    d_feature['shape1'] = _int64_feature([X.shape[1]])    
    d_feature['shape2'] = _int64_feature([X.shape[2]])

    features = tf.train.Features(feature=d_feature)
    example = tf.train.Example(features=features)
    serialized = example.SerializeToString()
    writer.write(serialized)
    
    if verbose:
        print("Writing {} done!".format(result_tf_file))

def data_to_tfrecords(X, Y, S,transform, spacing, file_path_prefix=None, verbose=True, debug=True):
           
    # Generate tfrecord writer
    result_tf_file = file_path_prefix + '.tfrecords'
           
    # Generate tfrecord writer
    result_tf_file = file_path_prefix + '.tfrecords'
    writer = tf.python_io.TFRecordWriter(result_tf_file)
    if verbose:
        print("Serializing example into {}".format(result_tf_file))
        
    # iterate over each sample,
    # and serialize it as ProtoBuf.
    
    d_feature = {}
    d_feature['X'] = _float_feature(X.flatten())
    d_feature['S'] = _int64_feature(S.flatten())

    if debug:
        print("**** X ****")
        print(X.shape, X.flatten().shape)
        print(X.dtype)
    for i, y in enumerate(Y):
        d_feature['Y_'+str(i)] = _float_feature(y.flatten())
        if debug:
            print("**** Y shape ****")
            print(y.shape, y.flatten().shape)
            print(y.dtype)

    d_feature['Transform'] = _float_feature(transform.flatten())
    d_feature['Spacing'] = _float_feature(spacing)
    #first axis is the channel dimension
    d_feature['shape0'] = _int64_feature([X.shape[0]])
    d_feature['shape1'] = _int64_feature([X.shape[1]])    
    d_feature['shape2'] = _int64_feature([X.shape[2]])

    features = tf.train.Features(feature=d_feature)
    example = tf.train.Example(features=features)
    serialized = example.SerializeToString()
    writer.write(serialized)
    
    if verbose:
        print("Writing {} done!".format(result_tf_file))

def max_pool_image(image, kernel=(2,2,2)):
    """
    Apply max pooling over image

    Args:
        image: pythong array
        kernel: max pool kernel size
    Returns:
        max_pooled: output python array
    """
    import skimage.measure
    max_pooled = skimage.measure.block_reduce(image, kernel, np.max)
    return max_pooled
