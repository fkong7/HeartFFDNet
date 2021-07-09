import tensorflow as tf
import functools
from augmentation import _augment
def _parse_function_all(mode):
    def __parse(example_proto):
        if mode=='img':
            features = {"X": tf.VarLenFeature(tf.float32),
              "shape0": tf.FixedLenFeature((), tf.int64),
              "shape1": tf.FixedLenFeature((), tf.int64),
              "shape2": tf.FixedLenFeature((), tf.int64),
              }
            parsed_features = tf.parse_single_example(example_proto, features)
            img = tf.sparse_tensor_to_dense(parsed_features["X"])
            depth = tf.cast(parsed_features["shape0"], tf.int32)
            height = tf.cast(parsed_features["shape1"], tf.int32)
            width = tf.cast(parsed_features["shape2"], tf.int32)
            img = tf.reshape(img, tf.stack([depth,height, width, 1]))
            return img
        elif mode=='seg':
            features = {"S": tf.VarLenFeature(tf.int64),
              "shape0": tf.FixedLenFeature((), tf.int64),
              "shape1": tf.FixedLenFeature((), tf.int64),
              "shape2": tf.FixedLenFeature((), tf.int64),
              }
            parsed_features = tf.parse_single_example(example_proto, features)
            seg = tf.sparse_tensor_to_dense(parsed_features["S"])
            depth = tf.cast(parsed_features["shape0"], tf.int32)
            height = tf.cast(parsed_features["shape1"], tf.int32)
            width = tf.cast(parsed_features["shape2"], tf.int32)
            seg = tf.reshape(seg, tf.stack([depth,height, width, 1]))
            return seg
        elif 'mesh' in mode:
            mesh_id = mode.split('_')[-1]
            features = {"Y_"+mesh_id: tf.VarLenFeature(tf.float32)
              }
            parsed_features = tf.parse_single_example(example_proto, features)
            mesh = tf.sparse_tensor_to_dense(parsed_features["Y_"+mesh_id])

            node_num = tf.cast(tf.shape(mesh)[0]/6, tf.int32)
            mesh = tf.reshape(mesh, tf.stack([node_num, 6 ]))
            return mesh
        elif mode=='transform':
            features = {"Transform": tf.VarLenFeature(tf.float32)}
            parsed_features = tf.parse_single_example(example_proto, features)
            transform = tf.sparse_tensor_to_dense(parsed_features["Transform"])
            transform = tf.reshape(transform, [4, 4])
            return transform
        elif mode=='spacing':
            features = {"Spacing": tf.VarLenFeature(tf.float32)}
            parsed_features = tf.parse_single_example(example_proto, features)
            spacing = tf.sparse_tensor_to_dense(parsed_features["Spacing"])
            spacing = tf.reshape(spacing, [3])
            return spacing
        elif mode=='center':
            features = {"center": tf.VarLenFeature(tf.float32)}
            parsed_features = tf.parse_single_example(example_proto, features)
            center = tf.sparse_tensor_to_dense(parsed_features["center"])
            center = tf.reshape(center, [6])
            return center 
        elif mode=='grid_pts':
            return tf.zeros((1,1))
        else:
            raise ValueError('invalid name')

    return __parse


def get_baseline_dataset(filenames, preproc_fn=functools.partial(_augment),
                         threads=1, 
                         batch_size=1,
                         mesh_ids = [2], # default is LV blood pool 2
                         shuffle=True,
                         if_seg=True,
                         num_block=1,
                         shuffle_buffer=128,
                         if_warp_im=False):           
  num_x = len(filenames)
  # Create a dataset from the filenames and labels
  files = tf.data.Dataset.from_tensor_slices(filenames)
  dataset = files.apply(tf.contrib.data.parallel_interleave(
    tf.data.TFRecordDataset, cycle_length=threads))
  # Map our preprocessing function to every element in our dataset, taking
  # advantage of multithreading
  dataset_img = dataset.map(_parse_function_all('img'))
  dataset_trans = dataset.map(_parse_function_all('transform'))
  dataset_spacing = dataset.map(_parse_function_all('spacing'))
  #dataset_center = dataset.map(_parse_function_all('center'))
  mesh_list = []
  for i in mesh_ids:
    dataset_mesh = dataset.map(_parse_function_all('mesh_'+str(i)))
    mesh_list.append(dataset_mesh)
  out_list = []
  grid_list = []
  im_list = []
  # Important node: we need to go through _parse_function_all instead of dataset from tensors tf.zeros((1,1))
  # otherwise it will generate the same dataset_center for all iterations
  for i in range(num_block):
      out_list += mesh_list
      dataset_grid = dataset.map(_parse_function_all('grid_pts'))
      grid_list.append(dataset_grid)
      if if_warp_im:
        dataset_im_diff = dataset.map(_parse_function_all('grid_pts'))
        im_list.append(dataset_im_diff)
  
  out_list = im_list + grid_list + out_list
  if if_seg:
    dataset_seg = dataset.map(_parse_function_all('seg'))
    out_list = [dataset_seg] + out_list

  dataset_input = tf.data.Dataset.zip((dataset_img, dataset_trans, dataset_spacing))
  dataset_output = tf.data.Dataset.zip(tuple(out_list))
  #dataset_output = tf.data.Dataset.zip((dataset_seg, tuple(mesh_list), tuple(mesh_list), tuple(mesh_list)))
  dataset = tf.data.Dataset.zip((dataset_input, dataset_output))
  dataset = dataset.map(preproc_fn)
  if shuffle:
    dataset = dataset.shuffle(shuffle_buffer)
  
  # It's necessary to repeat our data for all epochs 
  dataset = dataset.repeat().batch(batch_size, drop_remainder=True)
  #dataset = dataset.prefetch(buffer_size=batch_size)
  return dataset
