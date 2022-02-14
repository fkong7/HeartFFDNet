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
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
sys.path.append(os.path.join(os.path.dirname(__file__), "template"))
sys.path.append(os.path.join(os.path.dirname(__file__), "external"))
import tensorflow as tf
import SimpleITK as sitk 
from pre_process import *
from tensorflow.python.keras import backend as K
from model import HeartDeepFFD 
from data_loader import *
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from utils import *
from vtk_utils.vtk_utils import *
from make_control_grid import make_grid_vtk, make_grid, construct_bspline_volume,sparse_to_tuple, build_transform_matrix
import argparse
import pickle
import time
import scipy.sparse as sp
from scipy.spatial.distance import directed_hausdorff
def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image',  help='Name of the folder containing the image data')
    parser.add_argument('--mesh_dat',  help='Name of the .dat file containing mesh info')
    parser.add_argument('--model',  help='Name of the folder containing the trained model')
    parser.add_argument('--mesh_tmplt', help='Name of the finest mesh template')
    parser.add_argument('--attr',  help='Name of the image folder postfix')
    parser.add_argument('--output',  help='Name of the output folder')
    parser.add_argument('--modality', nargs='+', help='Name of the modality, mr, ct, split by space')
    parser.add_argument('--amplify_factor', type=float, default=1., help="amplify_factor of the predicted displacements")
    parser.add_argument('--size', type = int, nargs='+', help='Image dimensions')
    parser.add_argument('--mode', help='Test or validation (without or with ground truth label')
    parser.add_argument('--num_mesh', type=int, default=1, help='Number of meshes to train')
    parser.add_argument('--num_block', type=int,default=3, help='Number of graph conv block')
    parser.add_argument('--num_seg', type=int, default=8, help='Number of segmentation classes')
    parser.add_argument('--compare_seg', action='store_true', help='If to compare mesh with GT segmentation, otherwise compare with mesh')
    parser.add_argument('--d_weights', nargs='+', type=float, default=None, help='Weights to down-sample image first')
    parser.add_argument('--ras_spacing',nargs='+', type=float, default=None, help='Prediction spacing')
    parser.add_argument('--motion', action='store_true', help='If make prediction for all models.')
    parser.add_argument('--seg_id', default=[], type=int, nargs='+', help='List of segmentation ids to apply marching cube')
    parser.add_argument('--hidden_dim', type = int, default=128, help='Hidden dimension')
    parser.add_argument('--if_warp_im', action='store_true', help='If to deform image too')
    parser.add_argument('--if_swap_mesh', action='store_true', help='If to use a new mesh')
    args = parser.parse_args()
    return args

import csv
def write_scores(csv_path,scores): 
    with open(csv_path, 'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerow(('Dice', 'ASSD'))
        for i in range(len(scores)):
            writer.writerow(tuple(scores[i]))
            print(scores[i])
    writeFile.close()

class Prediction:
    #This class use the GCN model to predict mesh from 3D images
    def __init__(self, info, model_name, mesh_tmplt):
        self.heartFFD = HeartDeepFFD(**info)
        self.info = info
        self.model = self.heartFFD.build_keras('bspline')
        self.model_name = model_name
        self.model.load_weights(self.model_name)
        self.mesh_tmplt = mesh_tmplt
        self.amplify_factor = info['amplify_factor']
    
    def set_image_info(self, modality, image_fn, size, out_fn, mesh_fn=None, d_weights=None, write=False):
        self.modality = modality
        self.image_fn = image_fn
        self.image_vol = load_image_to_nifty(image_fn)
        self.origin = np.array(self.image_vol.GetOrigin())
        self.img_center = np.array(self.image_vol.TransformContinuousIndexToPhysicalPoint(np.array(self.image_vol.GetSize())/2.0))
        self.size = size
        self.out_fn = out_fn
        # down sample to investigate low resolution
        #self.image_vol = down_sample_spacing_with_factors(self.image_vol, factor=d_weights)
        if d_weights:
            self.image_vol = resample_spacing(self.image_vol, template_size = (384, 384, 384), order=1)[0]
            self.image_vol = down_sample_to_slice_thickness(self.image_vol, d_weights, order=0)
            if write:
                dir_name = os.path.dirname(self.out_fn)
                base_name = os.path.basename(self.out_fn)
                sitk.WriteImage(self.image_vol, os.path.join(dir_name, base_name+'_input_downsample.nii.gz'))
        self.image_vol = resample_spacing(self.image_vol, template_size = size, order=1)[0]
        if write:
            sitk.WriteImage(self.image_vol, os.path.join(dir_name, base_name+'_input_linear.nii.gz'))
        self.img_center2 = np.array(self.image_vol.TransformContinuousIndexToPhysicalPoint(np.array(self.image_vol.GetSize())/2.0))
        self.prediction = None
        self.mesh_fn = mesh_fn

    def mesh_prediction(self):
        img_vol = sitk.GetArrayFromImage(self.image_vol).transpose(2,1,0)
        img_vol = RescaleIntensity(img_vol,self.modality, [750, -750])
        self.original_shape = img_vol.shape
        transform = build_transform_matrix(self.image_vol)
        spacing = np.array(self.image_vol.GetSpacing())
        model_inputs = [np.expand_dims(np.expand_dims(img_vol, axis=-1), axis=0), np.expand_dims(transform, axis=0), np.expand_dims(spacing,axis=0)]
        start = time.time()
        prediction = self.model.predict(model_inputs)
        end = time.time()
        self.pred_time = end-start
        if self.heartFFD.num_seg > 0:
            prediction = prediction[1:]
        # remove control points output
        BLOCK_NUM = self.info['num_block']
        self.prediction_im = []
        grid_coords = tf.Session().run(self.info['feed_dict']['grid_coords']) 
        IMAGE_NUM = 0
        if self.info['if_warp_im']:
            IMAGE_NUM = BLOCK_NUM
            for i in range(BLOCK_NUM):
                curr_im_py = np.squeeze(prediction[i])
                grid_size = int(round(len(curr_im_py)**(1/3)))
                curr_im = sitk.GetImageFromArray(curr_im_py.reshape(grid_size, grid_size, grid_size).transpose(2,1,0))
                origin = list(np.min(grid_coords, axis=0).astype(float))
                curr_im.SetOrigin(list(np.min(grid_coords, axis=0).astype(float)))
                curr_im.SetDirection(list(np.eye(3).ravel().astype(float)))
                curr_im.SetSpacing(list(((np.max(grid_coords, axis=0)-np.min(grid_coords, axis=0))/(np.array(grid_size)-1)).astype(float)))
                self.prediction_im.append(curr_im)
        
        grid_mesh = []
        curr_grid  = None
        prediction_grid = prediction[IMAGE_NUM: BLOCK_NUM+IMAGE_NUM]
        for i in range(BLOCK_NUM):
            if self.info['if_output_grid']:
                curr_grid = np.squeeze(prediction_grid[i])
            else:
                if curr_grid is None:
                    curr_grid = grid_coords
                else:
                    #b_tf = tf.sparse.to_dense(self.info['feed_dict']['grid_upsample'][i-1])
                    #b = tf.Session().run(b_tf)
                    b = self.info['feed_dict']['grid_upsample'][i-1]
                    curr_grid = np.matmul(b, curr_grid)
                curr_grid += np.squeeze(prediction_grid[i]) * self.amplify_factor
            # Use the 4 lines below for projected prediction onto images
            grid_coords_out = curr_grid * np.array(self.size)
            grid_coords_out = np.concatenate((grid_coords_out, np.ones((grid_coords_out.shape[0],1))), axis=-1)
            grid_coords_out = np.matmul(transform, grid_coords_out.transpose()).transpose()[:,:3]
            grid_coords_out += self.img_center - self.img_center2
            grid_i = make_grid_vtk(grid_coords_out, False)
            # Use the line below for un-scaled prediction
            #grid_i = make_grid_vtk(curr_grid, False)
            grid_mesh.append(grid_i)
            
        self.prediction_grid = grid_mesh
        prediction_mesh = prediction[BLOCK_NUM+IMAGE_NUM:]
        num = len(prediction_mesh)//BLOCK_NUM
        self.prediction = []
        for i in range(BLOCK_NUM): # block number 
            mesh_i = vtk.vtkPolyData()
            mesh_i.DeepCopy(self.mesh_tmplt)
            pred_all = np.zeros((1, 0, 3))
            r_id = np.array([])
            for k in range(num):
                pred = prediction_mesh[i*num+k]
                pred_all = np.concatenate((pred_all, pred), axis=1)
                r_id = np.append(r_id, np.ones(pred.shape[1])*k)
            r_id_vtk = numpy_to_vtk(r_id)
            r_id_vtk.SetName('Ids')
            pred_all = np.squeeze(pred_all)
            # Use the line below for un-scaled prediction
            #pred_all /= np.array([128, 128, 128])
            # Use the 4 lines below for projected prediction onto images
            pred_all = pred_all * np.array(self.size)/np.array([128, 128, 128])
            pred_all = np.concatenate((pred_all,np.ones((pred_all.shape[0],1))), axis=-1)  
            pred_all = np.matmul(transform, pred_all.transpose()).transpose()[:,:3]
            pred_all = pred_all + self.img_center - self.img_center2
            mesh_i.GetPoints().SetData(numpy_to_vtk(pred_all))
            mesh_i.GetPointData().AddArray(r_id_vtk)
            self.prediction.append(mesh_i)
    
    def mesh_prediction_new_mesh(self, deform_mats):
        img_vol = sitk.GetArrayFromImage(self.image_vol).transpose(2,1,0)
        img_vol = RescaleIntensity(img_vol,self.modality, [750, -750])
        self.original_shape = img_vol.shape
        transform = build_transform_matrix(self.image_vol)
        spacing = np.array(self.image_vol.GetSpacing())
        model_inputs = [np.expand_dims(np.expand_dims(img_vol, axis=-1), axis=0), np.expand_dims(transform, axis=0), np.expand_dims(spacing,axis=0)]
        start = time.time()
        prediction = self.model.predict(model_inputs)
        end = time.time()
        self.pred_time = end-start
        if self.heartFFD.num_seg > 0:
            prediction = prediction[1:]
        # remove control points output
        BLOCK_NUM = self.info['num_block']
        self.prediction_im = []
        grid_coords = tf.Session().run(self.info['feed_dict']['grid_coords']) 
        IMAGE_NUM = 0
        if self.info['if_warp_im']:
            IMAGE_NUM = BLOCK_NUM
            for i in range(BLOCK_NUM):
                curr_im_py = np.squeeze(prediction[i])
                grid_size = int(round(len(curr_im_py)**(1/3)))
                curr_im = sitk.GetImageFromArray(curr_im_py.reshape(grid_size, grid_size, grid_size).transpose(2,1,0))
                origin = list(np.min(grid_coords, axis=0).astype(float))
                curr_im.SetOrigin(list(np.min(grid_coords, axis=0).astype(float)))
                curr_im.SetDirection(list(np.eye(3).ravel().astype(float)))
                curr_im.SetSpacing(list(((np.max(grid_coords, axis=0)-np.min(grid_coords, axis=0))/(np.array(grid_size)-1)).astype(float)))
                self.prediction_im.append(curr_im)
        
        grid_mesh = []
        self.prediction = []
        curr_grid  = None
        prediction_grid = prediction[IMAGE_NUM: BLOCK_NUM+IMAGE_NUM]
        for i in range(BLOCK_NUM):
            if curr_grid is None:
                curr_grid = grid_coords
                curr_grid += np.squeeze(prediction_grid[i]) * self.amplify_factor
            if i == 0:
                self.prediction.append(deform_mats[i].dot(curr_grid))
            else:
                self.prediction.append(self.prediction[-1] + deform_mats[i].dot(np.squeeze(prediction_grid[i]) * self.amplify_factor ))
            # Use the 4 lines below for projected prediction onto images
            grid_coords_out = curr_grid * np.array(self.size)
            grid_coords_out = np.concatenate((grid_coords_out, np.ones((grid_coords_out.shape[0],1))), axis=-1)
            grid_coords_out = np.matmul(transform, grid_coords_out.transpose()).transpose()[:,:3]
            grid_coords_out += self.img_center - self.img_center2
            grid_i = make_grid_vtk(grid_coords_out, False)
            # Use the line below for un-scaled prediction
            #grid_i = make_grid_vtk(curr_grid, False)
            grid_mesh.append(grid_i)
            
        self.prediction_grid = grid_mesh
        for i in range(len(self.prediction)): # block number 
            mesh_i = vtk.vtkPolyData()
            mesh_i.DeepCopy(self.mesh_tmplt)
            pred_all = self.prediction[i] * np.array([128, 128, 128])
            pred_all = np.concatenate((pred_all,np.ones((pred_all.shape[0],1))), axis=-1)  
            pred_all = np.matmul(transform, pred_all.transpose()).transpose()[:,:3]
            pred_all = pred_all + self.img_center - self.img_center2
            mesh_i.GetPoints().SetData(numpy_to_vtk(pred_all))
            self.prediction[i] = mesh_i
            
    def get_weights(self):
        self.model.load_weights(self.model_name)
        for layer in self.model.layers:
            print(layer.name, layer.get_config())
            weights = layer.get_weights()
            try:
                for w in weights:
                    print(np.max(w), np.min(w))
            except:
                print(weights)
       
    def evaluate_dice(self):
        print("Evaluating dice: ", self.image_fn, self.mesh_fn)
        ref_im = sitk.ReadImage(self.mesh_fn)
        ref_im, M = exportSitk2VTK(ref_im)
        ref_im_py = swapLabels_ori(vtk_to_numpy(ref_im.GetPointData().GetScalars()))
        pred_im_py = vtk_to_numpy(self.seg_result.GetPointData().GetScalars())
        dice_values = dice_score(pred_im_py, ref_im_py)
        return dice_values
    
    def evaluate_assd(self):
        def _get_assd(p_surf, g_surf):
            dist_fltr = vtk.vtkDistancePolyDataFilter()
            dist_fltr.SetInputData(1, p_surf)
            dist_fltr.SetInputData(0, g_surf)
            dist_fltr.SignedDistanceOff()
            dist_fltr.Update()
            distance_poly = vtk_to_numpy(dist_fltr.GetOutput().GetPointData().GetArray(0))
            return np.mean(distance_poly), dist_fltr.GetOutput()
        ref_im =  sitk.ReadImage(self.mesh_fn)
        ref_im = resample_spacing(ref_im, template_size=(256 , 256, 256), order=0)[0]
        ref_im, M = exportSitk2VTK(ref_im)
        ref_im_py = swapLabels_ori(vtk_to_numpy(ref_im.GetPointData().GetScalars()))
        ref_im.GetPointData().SetScalars(numpy_to_vtk(ref_im_py))
        
        dir_name = os.path.dirname(self.out_fn)
        base_name = os.path.basename(self.out_fn)
        pred_im = sitk.ReadImage(os.path.join(dir_name, base_name+'.nii.gz'))
        pred_im = resample_spacing(pred_im, template_size=(256,256,256), order=0)[0]
        pred_im, M = exportSitk2VTK(pred_im)
        pred_im_py = swapLabels_ori(vtk_to_numpy(pred_im.GetPointData().GetScalars()))
        pred_im.GetPointData().SetScalars(numpy_to_vtk(pred_im_py))

        ids = np.unique(ref_im_py)
        pred_poly_l = []
        dist_poly_l = []
        ref_poly_l = []
        dist = [0.]*len(ids)
        #evaluate hausdorff 
        haus = [0.]*len(ids)
        for index, i in enumerate(ids):
            if i==0:
                continue
            p_s = vtk_marching_cube(pred_im, 0, i)
            r_s = vtk_marching_cube(ref_im, 0, i)
            dist_ref2pred, d_ref2pred = _get_assd(p_s, r_s)
            dist_pred2ref, d_pred2ref = _get_assd(r_s, p_s)
            dist[index] = (dist_ref2pred+dist_pred2ref)*0.5

            haus_p2r = directed_hausdorff(vtk_to_numpy(p_s.GetPoints().GetData()), vtk_to_numpy(r_s.GetPoints().GetData()))
            haus_r2p = directed_hausdorff(vtk_to_numpy(r_s.GetPoints().GetData()), vtk_to_numpy(p_s.GetPoints().GetData()))
            haus[index] = max(haus_p2r, haus_r2p)
            pred_poly_l.append(p_s)
            dist_poly_l.append(d_pred2ref)
            ref_poly_l.append(r_s)
        dist_poly = appendPolyData(dist_poly_l)
        pred_poly = appendPolyData(pred_poly_l)
        ref_poly = appendPolyData(ref_poly_l)
        dist_r2p, _ = _get_assd(pred_poly, ref_poly)
        dist_p2r, _ = _get_assd(ref_poly, pred_poly)
        dist[0] = 0.5*(dist_r2p+dist_p2r)

        haus_p2r = directed_hausdorff(vtk_to_numpy(pred_poly.GetPoints().GetData()), vtk_to_numpy(ref_poly.GetPoints().GetData()))
        haus_r2p = directed_hausdorff(vtk_to_numpy(ref_poly.GetPoints().GetData()), vtk_to_numpy(pred_poly.GetPoints().GetData()))
        haus[0] = max(haus_p2r, haus_r2p)

        #dir_name = os.path.dirname(self.out_fn)
        #base_name = os.path.basename(self.out_fn)
        #fn = os.path.join(dir_name, 'distance_'+base_name+'.vtp')
        #write_vtk_polydata(dist_poly, fn)
        #fn = os.path.join(dir_name, 'pred_'+base_name+'.vtp')
        #write_vtk_polydata(pred_poly, fn)
        #fn = os.path.join(dir_name, 'ref_'+base_name+'.vtp')
        #write_vtk_polydata(ref_poly, fn)
        return dist, haus

    def write_prediction(self, seg_id, ras_spacing=None):
        #fn = '.'.join(self.out_fn.split(os.extsep, -1)[:-1])
        dir_name = os.path.dirname(self.out_fn)
        base_name = os.path.basename(self.out_fn)
        for i, pred in enumerate(self.prediction):
            fn_i =os.path.join(dir_name, 'block'+str(i)+'_'+base_name+'.vtp')
            write_vtk_polydata(pred, fn_i)
        for i, pred in enumerate(self.prediction_grid):
            fn_i =os.path.join(dir_name, 'block'+str(i)+'_'+base_name+'_grid.vtp')
            #write_vtk_polydata(pred, fn_i)
        if self.info['if_warp_im']:
            for i, pred in enumerate(self.prediction_im):
                fn_i =os.path.join(dir_name, 'block'+str(i)+'_'+base_name+'_im.nii.gz')
                sitk.WriteImage(pred, fn_i)
        _, ext = self.image_fn.split(os.extsep, 1)
        if ext == 'vti':
            ref_im = load_vtk_image(self.image_fn)
        else:
            im = sitk.ReadImage(self.image_fn)
            ref_im, M = exportSitk2VTK(im)
        if ras_spacing is not None:
            ref_im = vtkImageResample(ref_im, ras_spacing, 'NN')    
        out_im_py = np.zeros(vtk_to_numpy(ref_im.GetPointData().GetScalars()).shape)
        for s_id in seg_id:
            p = thresholdPolyData(self.prediction[-1], 'Ids', (s_id-1, s_id-1), 'point')
            pred_im = convertPolyDataToImageData(p, ref_im)
            pred_im_py = vtk_to_numpy(pred_im.GetPointData().GetScalars()) 
            if s_id == 7: # hard code for pulmonary artery
                mask = (pred_im_py==1) & (out_im_py==0) 
                out_im_py[mask] = s_id 
            else:
                out_im_py[pred_im_py==1] = s_id
        ref_im.GetPointData().SetScalars(numpy_to_vtk(out_im_py))
        self.seg_result = ref_im
        if ext == 'vti':
            write_vtk_image(ref_im, os.path.join(dir_name, base_name+'.vti'))
        else:
            vtk_write_mask_as_nifty(ref_im, M, self.image_fn, os.path.join(dir_name, base_name+'.nii.gz'))

if __name__ == '__main__':
    args = parse()
    try:
        os.makedirs(args.output)
    except Exception as e: print(e)
    import time
    start = time.time()
    #load image filenames
    BATCH_SIZE = 1
    pkl = pickle.load(open(args.mesh_dat, 'rb'))
    mesh_tmplt = load_vtk_mesh(args.mesh_tmplt)
    deform_mats = [None]*len(pkl['ffd_matrix_mesh'])
    for i, mat in enumerate(pkl['ffd_matrix_mesh']):
        try:
            ctrl_pts = pkl['grid_coords_stored'][i] #Later on, store grid coordinates
        except Exception as e:
            min_bound, max_bound = np.min(pkl['grid_coords'], axis=0)-0.001, np.max(pkl['grid_coords'], axis=0)+0.001
            num_pts = mat[-1][-1]
            num_pts = int(round(num_pts**(1/3)))
            grid = make_grid(num_pts, (min_bound, max_bound))
            ctrl_pts = vtk_to_numpy(grid.GetPoints().GetData())
            #write_vtk_polydata(grid, os.path.join(args.output, 'grid_{}.vtp'.format(num_pts)))
        mesh_coords = vtk_to_numpy(mesh_tmplt.GetPoints().GetData())
        new_matrix = construct_bspline_volume(ctrl_pts, mesh_coords, (min_bound, max_bound), order=3)
        deform_mats[i] = new_matrix
        # debug 
    mesh_info = construct_feed_dict(pkl)
    info = {'batch_size': BATCH_SIZE,
            'input_size': (args.size[0], args.size[1], args.size[2], 1),
            'hidden_dim': args.hidden_dim,
            'feed_dict': mesh_info,
            'num_mesh': args.num_mesh,
            'num_seg': args.num_seg,
            'num_block': args.num_block, 
            'amplify_factor': args.amplify_factor, 
            'if_warp_im': args.if_warp_im,
            'if_output_grid': True
            }
    filenames = {}
    extensions = ['nii', 'nii.gz', 'vti']
    predict = Prediction(info, args.model, mesh_tmplt)
    #predict.get_weights()
    for m in args.modality:
        if args.compare_seg:
            x_filenames, y_filenames = [], []
            for ext in extensions:
                im_loader = DataLoader(m, args.image, fn=args.attr, fn_mask=None if args.mode=='test' else args.attr+'_masks', ext='*.'+ext, ext_out='*.'+ext)
                x_fns_temp, y_fns_temp = im_loader.load_datafiles()
                x_filenames += x_fns_temp
                y_filenames += y_fns_temp
        else:
            x_filenames, y_filenames = [], []
            for ext in extensions:
                im_loader = DataLoader(m, args.image, fn=args.attr, fn_mask=None if args.mode=='test' else args.attr+'_masks', ext='*.'+ext, ext_out='*.'+ext)
                x_fns_temp, _= im_loader.load_datafiles()
                x_filenames += x_fns_temp
            im_loader = DataLoader(m, args.image, fn=args.attr, fn_mask=args.attr+'_seg', ext='*.vtp', ext_out='*.vtp')
            _, y_filenames = im_loader.load_datafiles()
            im_loader = DataLoader(m, args.image, fn=args.attr, fn_mask=args.attr+'_seg', ext='*.vtk', ext_out='*.vtk')
            _, y_filenames2 = im_loader.load_datafiles()
            y_filenames += y_filenames2
        x_filenames = natural_sort(x_filenames)
        try:
            y_filenames = natural_sort(y_filenames)
        except: pass
        score_list = []
        assd_list = []
        haus_list = []
        time_list = []
        time_list2 = []
        for i in range(len(x_filenames)):
            #set up models
            print("processing "+x_filenames[i])
            start2 = time.time()
            if args.motion:
                out_fn = os.path.basename(x_filenames[i]).split('.')[0]+'_'+'epoch_'+ str(mdl_id)
            else:    
                out_fn = os.path.basename(x_filenames[i]).split('.')[0]
            predict.set_image_info(m, x_filenames[i], args.size, os.path.join(args.output, out_fn), y_filenames[i], d_weights=args.d_weights, write=False)
            #predict.get_weights()
            if args.if_swap_mesh:
                predict.mesh_prediction_new_mesh(deform_mats)
            else:
                predict.mesh_prediction()
            predict.write_prediction(args.seg_id, args.ras_spacing)
            time_list.append(predict.pred_time)
            end2 = time.time()
            time_list2.append(end2-start2)
            if y_filenames[i] is not None:
                #score_list.append(predict.evaluate(args.seg_id,out_fn ))
                score_list.append(predict.evaluate_dice())
                assd, haus = predict.evaluate_assd()
                assd_list.append(assd)
                haus_list.append(haus)
                #metric_names = predict.get_metric_names
        if len(score_list) >0:
            csv_path = os.path.join(args.output, '%s_test.csv' % m)
            csv_path_assd = os.path.join(args.output, '%s_test_assd.csv' % m)
            csv_path_haus = os.path.join(args.output, '%s_test_haus.csv' % m)
            write_scores(csv_path, score_list)
            write_scores(csv_path_assd, assd_list)
            write_scores(csv_path_haus, haus_list)

    end = time.time()
    print("Total time spent: ", end-start)
    print("Avg pred time ", np.mean(time_list)) 
    print("Avg generation time", np.mean(time_list2))
