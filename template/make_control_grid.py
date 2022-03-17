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
sys.path.append(os.path.join(os.path.dirname(__file__), '../external'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import numpy as np
#np.random.seed(42) 
from vtk_utils.vtk_utils import *
from pre_process import *
import argparse 
from datetime import datetime
import scipy.sparse as sp
import pickle
from scipy.sparse.linalg.eigen.arpack import eigsh

def build_transform_matrix(image):
    matrix = np.eye(4)
    matrix[:-1,:-1] = np.matmul(np.reshape(image.GetDirection(), (3,3)), np.diag(image.GetSpacing()))
    matrix[:-1,-1] = np.array(image.GetOrigin())
    return matrix 

def map_polydata_coords(poly, displacement, transform, size):
    coords = vtk_to_numpy(poly.GetPoints().GetData())
    coords += displacement
    coords = np.concatenate((coords,np.ones((coords.shape[0],1))), axis=-1) 
    coords = np.matmul(np.linalg.inv(transform), coords.transpose()).transpose()[:,:3]
    coords /= np.array(size)
    return coords

def transform_polydata(poly, displacement, transform, size):
    coords = map_polydata_coords(poly, displacement, transform, size)
    poly.GetPoints().SetData(numpy_to_vtk(coords))
    return poly

def get_image_patch(image_py, coords):
    """
    return a patch of the image defined under coords, the coords should be in [0,1]R^3
    """
    dim_x, dim_y, dim_z = image_py.shape
    indices = coords * np.array([[dim_x, dim_y, dim_z]])
    x1 = np.floor(indices[:,0]).astype(int)
    y1 = np.floor(indices[:,1]).astype(int)  
    z1 = np.floor(indices[:,2]).astype(int)  
    x2 = np.ceil(indices[:,0]).astype(int)  
    y2 = np.ceil(indices[:,1]).astype(int)  
    z2 = np.ceil(indices[:,2]).astype(int)  
    q11 = image_py[x1, y1, z1]
    q21 = image_py[x2, y1, z1]
    q12 = image_py[x1, y2, z1]
    q22 = image_py[x2, y2, z1]

    wx = indices[:, 0] - x1
    wx2 = x2 - indices[:, 0]
    lerp_x1 = q21 * wx + q11 * wx2
    lerp_x2 = q12 * wx + q22 * wx2

    wy = indices[:, 1] - y1
    wy2 = y2 - indices[:, 1]
    lerp_y1 = lerp_x2 * wy + lerp_x1 * wy2

    q112 = image_py[x1, y1, z2]
    q212 = image_py[x2, y1, z2]
    q122 = image_py[x1, y2, z2]
    q222 = image_py[x2, y2, z2]

    lerp_x12 = q212 * wx + q112 * wx2
    lerp_x22 = q122 * wx + q222 * wx2
    lerp_y12 = lerp_x22 * wy + lerp_x12 * wy2

    wz = indices[:, 2] - z1
    wz2 = z2 - indices[:,2]
    lerp_z = lerp_y12 * wz + lerp_y1 * wz2
    return lerp_z


def make_grid_vtk(ctrl_points, diagonal=True):
    # assume equal number of control points along each dim
    num_pts = int(round(len(ctrl_points)**(1/3)))
    
    # points
    grid = vtk.vtkPolyData()
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_to_vtk(ctrl_points))
    grid.SetPoints(vtk_points)

    # edges
    lines = vtk.vtkCellArray()
    for i in range(num_pts):
        for j in range(num_pts):
            for k in range(num_pts-1):
                id1 = i*num_pts*num_pts+j*num_pts +k 
                ids = []
                ids.append(i*num_pts*num_pts+j*num_pts +k+1)
                if diagonal:
                    if j<num_pts-1:
                        ids.append(i*num_pts*num_pts+(j+1)*num_pts +k+1)
                        if i < num_pts-1:
                            ids.append((i+1)*num_pts*num_pts+(j+1)*num_pts +k+1)
                        if i >0:
                            ids.append((i-1)*num_pts*num_pts+(j+1)*num_pts +k+1)
                    if j>0:
                        ids.append(i*num_pts*num_pts+(j-1)*num_pts +k+1)
                        if i < num_pts-1:
                            ids.append((i+1)*num_pts*num_pts+(j-1)*num_pts +k+1)
                        if i >0:
                            ids.append((i-1)*num_pts*num_pts+(j-1)*num_pts +k+1)
                        #if i<num_pts-1:
                        #    ids.append((i+1)*num_pts*num_pts+(j+1)*num_pts +k)
                for id_p in ids:
                    line = vtk.vtkLine()
                    line.GetPointIds().SetId(0, id1)
                    line.GetPointIds().SetId(1, id_p)
                    lines.InsertNextCell(line)
    for i in range(num_pts):
        for j in range(num_pts-1):
            for k in range(num_pts):
                id1 = i*num_pts*num_pts+j*num_pts +k 
                ids = []
                ids.append(i*num_pts*num_pts+(j+1)*num_pts +k)
                if diagonal:
                    if i<num_pts-1:
                        ids.append((i+1)*num_pts*num_pts+(j+1)*num_pts +k)
                    if i>0:
                        ids.append((i-1)*num_pts*num_pts+(j+1)*num_pts +k)
                for id_p in ids:
                    line = vtk.vtkLine()
                    line.GetPointIds().SetId(0, id1)
                    line.GetPointIds().SetId(1, id_p)
                    lines.InsertNextCell(line)
    for i in range(num_pts-1):
        for j in range(num_pts):
            for k in range(num_pts):
                id1 = i*num_pts*num_pts+j*num_pts +k 
                ids = []
                ids.append((i+1)*num_pts*num_pts+j*num_pts +k)
                if diagonal:
                    if k<num_pts-1:
                        ids.append((i+1)*num_pts*num_pts+j*num_pts +k+1)
                    if k>0:
                        ids.append((i+1)*num_pts*num_pts+j*num_pts +k-1)
                for id_p in ids:
                    line = vtk.vtkLine()
                    line.GetPointIds().SetId(0, id1)
                    line.GetPointIds().SetId(1, id_p)
                    lines.InsertNextCell(line)
    grid.SetLines(lines)
    return grid
    
def make_grid(num_pts, bounds, diagonal=True):
    # compute bounding box of the template
    min_bound, max_bound  =  bounds
    # create control points
    x = np.linspace(min_bound[0], max_bound[0], num_pts, endpoint=True)
    y = np.linspace(min_bound[1], max_bound[1], num_pts, endpoint=True)
    z = np.linspace(min_bound[2], max_bound[2], num_pts, endpoint=True)

    # create vtk polydata 
    u, v, w = np.meshgrid(x, y, z, indexing='ij')
    coords = np.column_stack((u.flatten(), v.flatten(), w.flatten()))
    grid = make_grid_vtk(coords, diagonal)

    #write_vtk_polydata(grid, os.path.join(os.path.dirname(__file__), 'grid_pts{}.vtk'.format(num_pts)))
    return grid

    
def load_geometry_from_file(fn, target_node_num):
    template = load_vtk_mesh(fn)
    try:
        region_ids = np.unique(vtk_to_numpy(template.GetCellData().GetArray('Scalars_'))).astype(int)
    except:
        region_ids = np.unique(vtk_to_numpy(template.GetPointData().GetArray('RegionId'))).astype(int)
    print("Unique ids of template mesh: ", region_ids)
    struct_list = []
    node_list = [0]
    total_node = 0
    face_list = []
    region_id = []
    for i in region_ids:
        poly_i = thresholdPolyData(template, 'Scalars_', (i, i),'cell')
        if poly_i.GetNumberOfPoints() == 0:
            poly_i = thresholdPolyData(template, 'RegionId', (i, i),'point')
        num_pts = poly_i.GetNumberOfPoints()
        rate = max(0., 1. - float(target_node_num)/num_pts)
        print("Target reduction rate of structure: ", i, target_node_num, num_pts, rate)
        poly_i = decimation(poly_i, rate)
        total_node += poly_i.GetNumberOfPoints()
        node_list.append(total_node)
        struct_list.append(poly_i)
        cells = vtk_to_numpy(poly_i.GetPolys().GetData()) 
        cells = cells.reshape(poly_i.GetNumberOfCells(), 4)
        cells = cells[:,1:]
        region_id += list(np.ones(poly_i.GetNumberOfCells())*i)
        face_list.append(cells)
    template_deci = appendPolyData(struct_list)
    
    region_id_vtk = numpy_to_vtk(region_id)
    region_id_vtk.SetName('Scalars_')
    template_deci.GetCellData().AddArray(region_id_vtk)
    return template_deci, node_list, face_list

def process_template(template_fn, target_node_num=None, template_im_fn=None, ref_template_fn=None):
    if target_node_num is None:
        template = load_vtk_mesh(template_fn)
        node_list = [template.GetNumberOfPoints()]
        face_list = vtk_to_numpy(template.GetPolys().GetData()).reshape(template.GetNumberOfCells(), 4)[:, 1:]
    else:
        template, node_list, face_list  = load_geometry_from_file(template_fn, target_node_num)
    if template_im_fn is None:
        coords = vtk_to_numpy(template.GetPoints().GetData())
    else:
        SIZE = (128, 128, 128)
        imgVol_o = sitk.ReadImage(template_im_fn)
        img_center = np.array(imgVol_o.TransformContinuousIndexToPhysicalPoint(np.array(imgVol_o.GetSize())/2.0))
        imgVol = resample_spacing(imgVol_o, template_size=SIZE, order=1)[0]  # numpy array
        img_center2 = np.array(imgVol.TransformContinuousIndexToPhysicalPoint(np.array(imgVol.GetSize())/2.0))
        transform = build_transform_matrix(imgVol)
        template  = transform_polydata(template, img_center2-img_center, transform, SIZE)
        coords = vtk_to_numpy(template.GetPoints().GetData())
    #write_vtk_polydata(template, os.path.join(os.path.dirname(__file__), datetime.now().strftime("%m_%d_%Y_%H_%M_%S")+'_template_'+os.path.basename(template_fn)))
    #write_vtk_polydata(template, os.path.join(os.path.dirname(__file__), '../examples/template_with_veins_normalized.vtp'))
    if ref_template_fn is not None:
        bounds = (np.min(ref_coords, axis=0), np.max(ref_coords, axis=0))
    else:
        bounds = (np.min(coords, axis=0), np.max(coords, axis=0))
    return template, node_list, face_list, bounds


from math import factorial

def comb(n, k):
    return factorial(n) / factorial(k) / factorial(n - k)

def ffd(ctrl_pts, tmplt_coords, bounds):
    '''
    Ctrl points or d_cntrl points should be in world coordinates
    Tmple_coords is in world coordinates and will be normalized to grid coordinates
    '''
    min_bound, max_bound = bounds
    tmplt_coords = tmplt_coords - np.expand_dims(min_bound, axis=0)
    tmplt_coords /= np.expand_dims(max_bound - min_bound, axis=0)
    num_pts = int(round(len(ctrl_pts)**(1/3)))
    # Bernstein tensor
    B = []
    for i in range(num_pts):
        for j in range(num_pts):
            for k in range(num_pts):
                coeff = comb(num_pts-1, k) * comb(num_pts-1, j) * comb(num_pts-1, i)
                b_list = coeff * ((1 - tmplt_coords[:,0]) ** (num_pts-1 - i)) * (tmplt_coords[:,0] ** i) \
                        * ((1 - tmplt_coords[:,1]) ** (num_pts-1 - j)) * (tmplt_coords[:,1] ** j)\
                        * ((1 - tmplt_coords[:,2]) ** (num_pts-1 - k)) * (tmplt_coords[:,2] ** k)
                B.append(b_list)
    B = np.stack(B, axis=1)
    B[B<1e-5] = 0.
    s_B = sp.csr_matrix(B, copy=True)
    print("Number of elements in grid matrix: ", len(sparse_to_tuple(s_B)[1]))
    output = s_B.dot(ctrl_pts)
    #output = np.matmul(B, ctrl_pts)
    return output, sparse_to_tuple(s_B)

def construct_bspline_volume(ctrl_pts, tmplt_coords, bounds, order=3):
    min_bound, max_bound = bounds
    num_pts = int(round(len(ctrl_pts)**(1/3)))

    # create knot vectors
    u, v, w = [], [], []
    for i in range(num_pts+order+1):
        coeff = min(max(0, i-order), num_pts-order)
        u.append(min_bound[0] + coeff*(max_bound[0]-min_bound[0])/(num_pts-order))
        v.append(min_bound[1] + coeff*(max_bound[1]-min_bound[1])/(num_pts-order))
        w.append(min_bound[2] + coeff*(max_bound[2]-min_bound[2])/(num_pts-order))
    #print("knots: ", u)
    #print("knots: ", v)
    #print("knots: ", w)
    return construct_bspline_matrix(ctrl_pts, tmplt_coords, u, v, w, order)

def construct_bspline_matrix(ctrl_pts, tmplt_coords, u, v, w, order=3):
    def _compute_basis(x, t, i, p):
        if p == 0:
            #b = np.where((x >= t[i]-1e-5) & (x <= t[i+1]+1e-5), 1., 0.)
            #b = np.where((x >= t[i]) & (x <= t[i+1]), 1., 0.)
            b = np.where((x >= t[i]-1e-4) & (x <= t[i+1]+1e-4), 1., 0.)
            return b
        seg_i = t[i+p] - t[i]
        seg_ip1 = (t[i+p+1] - t[i+1])
        if np.isclose(seg_i, 0.):
            left = np.zeros(x.shape)
        else:
            left = (x - t[i])/seg_i * _compute_basis(x, t, i, p-1)
        if np.isclose(seg_ip1, 0.):
            right = np.zeros(x.shape)
        else:
            right = (t[i+p+1] - x)/(t[i+p+1] - t[i+1]) * _compute_basis(x, t, i+1, p-1)
        b = left + right
        return b
    num_pts = int(round(len(ctrl_pts)**(1/3)))
    B = []
    B = []
    for i in range(num_pts):
        for j in range(num_pts):
            for k in range(num_pts):
                basis_u = _compute_basis(tmplt_coords[:,0], u, i, order)
                basis_v = _compute_basis(tmplt_coords[:,1], v, j, order)
                basis_w = _compute_basis(tmplt_coords[:,2], w, k, order)
                b_list = basis_u * basis_v * basis_w
                B.append(b_list)
    B = np.stack(B, axis=1)
    if np.any(np.sum(B, axis=-1)==0):
        raise RuntimeError("NaN in the B spline matrix!.")
    #np.set_printoptions(threshold=np.inf)
    #print(B)
    B /= np.sum(B, axis=-1, keepdims=True)
    B[B<1e-5] = 0.
    B[np.isnan(B)] = 0.
    #print("Check NaN: ", np.any(np.isnan(B)))
    #print("Check Inf: ", np.any(np.isinf(B)))
    #print(B)
    s_B = sp.csr_matrix(B, copy=True)
    print("Number of elements in grid matrix: ", len(sparse_to_tuple(s_B)[1]))
    return s_B

def bspline(Basis_matrix, curr_grid, order=3):
    if type(Basis_matrix)==tuple:
        Basis_matrix = sp.csr_matrix((Basis_matrix[1], (Basis_matrix[0][:,0], Basis_matrix[0][:,1])), shape=Basis_matrix[-1])
    output = Basis_matrix.dot(curr_grid)
    return output


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)

def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))
    
    return sparse_to_tuple(t_k)

def transition_matrix_for_multi_level_grid(grid1, grid2, inverse=False):
    """
    build a matrix B such that grid2 = B grid1
    we assume grid2 is denser than grid1
    if inverse, we can compute the left inverse (B^TB)^-1B^T
    """
    grid1_min, grid1_max = np.min(grid1, axis=0, keepdims=True), np.max(grid1, axis=0, keepdims=True)
    grid2_min, grid2_max = np.min(grid2, axis=0, keepdims=True), np.max(grid2, axis=0, keepdims=True)
    grid2_nrmed = (grid2 - grid2_min)/grid2_max
    grid1_nrmed = (grid1 - grid1_min)/grid1_max
    # find steps
    x_step = np.unique(grid1_nrmed[:, 0])
    y_step = np.unique(grid1_nrmed[:, 1])
    z_step = np.unique(grid1_nrmed[:, 2])
    num_x, num_y, num_z = len(x_step), len(y_step), len(z_step)

    steps = [x_step[1]-x_step[0], y_step[1]-y_step[0], z_step[1]-z_step[0]]
    indices = np.round(grid2_nrmed/np.array(steps), decimals=5)
    B = np.zeros((grid2_nrmed.shape[0], grid1_nrmed.shape[0]))
    ind_f = np.floor(indices)
    ind_c = np.ceil(indices)
    ind_f = np.where(ind_f==ind_c, ind_f-1., ind_f)
    mask = ind_f<0
    ind_f[mask] = 0.
    ind_c[mask] =1.
    ind_corners = [ind_f, ind_c]
    w_f = ind_c - indices
    w_c = indices - ind_f
    weight_corners = [w_f, w_c]
    for i in range(len(ind_corners)):
        x_comp = ind_corners[i][:,0]*num_y*num_z
        for j in range(len(ind_corners)):
            y_comp = ind_corners[j][:,1]*num_z
            for k in range(len(ind_corners)):
                z_comp = ind_corners[k][:,2]
                ind = x_comp + y_comp + z_comp
                weight = weight_corners[i][:,0]*weight_corners[j][:,1]*weight_corners[k][:,2]
                B[range(grid2_nrmed.shape[0]), ind.astype(int)] = weight
    # debug:
    test = np.sum(np.matmul(B, grid1) - grid2)
    print("Test error: ", test)
    if inverse:
        inv = np.linalg.inv(np.matmul(B.transpose(), B))
        B = np.matmul(inv, B.transpose())
        test = np.sum(np.matmul(B, grid2) - grid1)
        print("Inverse test error: ", test)
        return B
    else:
        s_B = sp.csr_matrix(B, copy=True)
        print("Number of elements in upsample matrix: ", len(sparse_to_tuple(s_B)[1]))
        return sparse_to_tuple(s_B)

def transition_matrix_for_multi_level_grid_gaussion(grid1, grid2):
    """
    build a matrix B such that grid2 = B grid1
    we assume grid2 is denser than grid1
    if inverse, we can compute the left inverse (B^TB)^-1B^T
    """
    grid1_min, grid1_max = np.min(grid1, axis=0, keepdims=True), np.max(grid1, axis=0, keepdims=True)
    grid2_min, grid2_max = np.min(grid2, axis=0, keepdims=True), np.max(grid2, axis=0, keepdims=True)
    grid2_nrmed = (grid2 - grid2_min)/grid2_max
    grid1_nrmed = (grid1 - grid1_min)/grid1_max
    # find steps
    x_step = np.unique(grid2_nrmed[:, 0])
    y_step = np.unique(grid2_nrmed[:, 1])
    z_step = np.unique(grid2_nrmed[:, 2])

    num_x, num_y, num_z = len(x_step), len(y_step), len(z_step)
    B = np.zeros((grid2.shape[0], grid1.shape[0]))

    #assum the grid distribution is uniform
    x_space = np.mean(x_step[1:] - x_step[:-1])/3.
    y_space = np.mean(y_step[1:] - y_step[:-1])/3.
    z_space = np.mean(z_step[1:] - z_step[:-1])/3.
    co_var = np.diag([x_space, y_space, z_space])
    inv_co_var = np.linalg.inv(co_var)
    for i in range(grid2.shape[0]):
        curr_pt = np.expand_dims(grid2[i, :], axis=0)
        prob = np.sum(np.matmul(grid1-curr_pt, inv_co_var)*(grid1-curr_pt), axis=-1)
        B[i,:] = np.squeeze(prob)
    B = np.exp(-0.5*B)/np.sqrt((2*np.pi)**3*np.linalg.det(co_var))
    B = B/np.max(B, axis=-1, keepdims=True)
    rej = B * np.random.rand(*B.shape)
    sort_rej = np.argsort(rej, axis=-1)
    thres = np.expand_dims(rej[range(B.shape[0]), sort_rej[:, -16]], axis=-1)
    B[np.less(rej, thres)] = 0.
    B = B/np.sum(B, axis=-1, keepdims=True)
    print("CHECK min, max: ", np.min(B[B>0.]), np.mean(B[B>0.]), np.max(B))
    #B[B<1e-5] = 0.
    s_B = sp.csr_matrix(B, copy=True)
    print("Number of elements in down-sample matrix: ", len(sparse_to_tuple(s_B)[1]))
    return sparse_to_tuple(s_B)
    #return B

def test_bspline(grid, template, bounds, B_matrix=None):
    coords = vtk_to_numpy(template.GetPoints().GetData())
    # deform
    # Control pts
    ctrl_pts = vtk_to_numpy(grid.GetPoints().GetData())
    if B_matrix is None:
        num_pts = int(round(len(ctrl_pts)**(1/3)))
        B_matrix = construct_bspline_volume(ctrl_pts, coords, bounds)
        # random deformation
        #cntr = np.mean(ctrl_pts, axis=0)
        #d_grid = np.random.rand(*ctrl_pts.shape)*0.1
        # deform center plane
        c_pt_min = np.min(ctrl_pts, axis=0, keepdims=True)
        c_pt_max = np.max(ctrl_pts, axis=0, keepdims=True)
        d_grid = + (ctrl_pts-c_pt_min) * ((ctrl_pts-c_pt_min) - c_pt_max) 
        # translate
        #d_grid = np.ones(ctrl_pts.shape)* np.random.rand(1,3)
        # zeros
        #d_grid = np.zeros(ctrl_pts.shape)

        ctrl_pts += d_grid
        grid_deform = vtk.vtkPolyData()
        grid_deform.DeepCopy(grid)
        grid_deform.GetPoints().SetData(numpy_to_vtk(ctrl_pts))
        write_vtk_polydata(grid_deform, os.path.join(os.path.dirname(__file__), 'deformed_bspline_grid{}.vtp'.format(num_pts)))

    # Convert template into control point grid
    output =  bspline(B_matrix, ctrl_pts)
    #output = coords + bspline(B_matrix, d_grid)
    output_poly = vtk.vtkPolyData()
    output_poly.DeepCopy(template)
    output_poly.GetPoints().SetData(numpy_to_vtk(output))

    write_vtk_polydata(output_poly, os.path.join(os.path.dirname(__file__), 'deformed_bspline.vtp'))


def test_ffd(grid, template, bounds):
    coords = vtk_to_numpy(template.GetPoints().GetData())
    # deform
    # Control pts
    ctrl_pts = vtk_to_numpy(grid.GetPoints().GetData())
    num_pts = int(round(len(ctrl_pts)**(1/3)))
    # random deformation
    #cntr = np.mean(ctrl_pts, axis=0)
    #d_grid = np.random.rand(*ctrl_pts.shape)*0.1
    #d_grid -= np.mean(d_grid, axis=0) 

    #d_grid = np.zeros(ctrl_pts.shape)
    #d_grid[0, :] = np.random.rand(3) 
    #ctrl_pts += d_grid

    # deform center planes
    c_pt_min = np.min(ctrl_pts, axis=0, keepdims=True)
    c_pt_max = np.max(ctrl_pts, axis=0, keepdims=True)
    d_grid = + (ctrl_pts-c_pt_min) * ((ctrl_pts-c_pt_min) - c_pt_max) 
    # translation
    #rand_add = np.random.rand(1, 3)*10
    #ctrl_pts += rand_add
    
    grid_deform = vtk.vtkPolyData()
    grid_deform.DeepCopy(grid)
    grid_deform.GetPoints().SetData(numpy_to_vtk(ctrl_pts))
    write_vtk_polydata(grid_deform, os.path.join(os.path.dirname(__file__), 'deformed_grid{}.vtp'.format(num_pts)))
    
    # Convert template into control point grid
    #output = ffd(ctrl_pts, coords_nrm)
    output =  coords + ffd(d_grid, coords, bounds)[0]
    output_poly = vtk.vtkPolyData()
    output_poly.DeepCopy(template)
    output_poly.GetPoints().SetData(numpy_to_vtk(output))

    write_vtk_polydata(output_poly, os.path.join(os.path.dirname(__file__), 'deformed.vtp'))

def make_dat_file_ffd(num_pts, template_processed, mode, order=3, image_fn=None, write=True):
    from scipy import sparse
    template, node_list, faces, bounds = template_processed
    coords = vtk_to_numpy(template.GetPoints().GetData())
    info = {'struct_node_ids':node_list, 'tmplt_faces': faces, 'grid_coords': None, 
            'ffd_matrix_mesh': None, 'ffd_matrix_grid': None, 
            'support': None, 'ffd_matrix_image': None}
    info['tmplt_coords'] = coords
    
    if image_fn is not None:
        GRID_SIZE = 32
        IMAGE_SIZE = (128, 128, 128)
        image = sitk.ReadImage(image_fn)
        imgVol = resample_spacing(image, template_size=IMAGE_SIZE, order=1)[0]
        img_py = RescaleIntensity(sitk.GetArrayFromImage(imgVol).transpose(2,1,0), 'ct', [750, -750])
        image_grid = make_grid(GRID_SIZE, bounds)
        info['image_data'] = get_image_patch(img_py, image_grid_coords)

    grid = make_grid(num_pts, bounds)
    grid_sample = make_grid(num_pts, (bounds[0]+0.01, bounds[1]-0.01))
    sample_pts = vtk_to_numpy(grid_sample.GetPoints().GetData())
    ctrl_pts = vtk_to_numpy(grid.GetPoints().GetData())
    info['grid_coords'] = ctrl_pts
    #build adjacency matrix for the grid
    adj = np.zeros((len(ctrl_pts), len(ctrl_pts)))
    lines = vtk_to_numpy(grid.GetLines().GetData()).reshape(grid.GetNumberOfLines(), 3)
    lines = lines[:, 1:]
    for i in range(lines.shape[0]):
        adj[lines[i, 0], lines[i, 1]] = 1
        adj[lines[i, 1], lines[i, 0]] = 1
    #print("Degree: ", np.sum(adj, axis=-1))
    #print(adj.shape)
    adj_s = sparse.csr_matrix(adj)
    cheb = chebyshev_polynomials(adj_s,1)
    info['support'] = cheb 
    if mode == 'ffd':
        info['ffd_matrix_mesh'] = ffd(ctrl_pts, coords, bounds)[-1]
        info['ffd_matrix_grid'] = ffd(ctrl_pts, ctrl_pts, bounds)[-1]
    elif mode=='bspline':
        info['ffd_matrix_mesh'] = sparse_to_tuple(construct_bspline_volume(ctrl_pts, coords, bounds))
        info['ffd_matrix_grid'] = sparse_to_tuple(construct_bspline_volume(ctrl_pts, sample_pts, bounds))
    else:
        raise NotImplementedError
    # FFD for the image for now only on last block
    if image_fn is not None:
        if mode == 'ffd':
            info['ffd_matrix_image'] = ffd(ctrl_pts, image_grid_coords, bounds)[-1]
        elif mode=='bspline':
            mesh_b  = construct_bspline_volume(ctrl_pts, image_grid_coords,  bounds)
            info['ffd_matrix_image'] = sparse_to_tuple(mesh_b)
        else:
            raise NotImplementedError
        # debug
        debug_im = sitk.GetImageFromArray(info['image_data'].reshape(GRID_SIZE, GRID_SIZE, GRID_SIZE).transpose(2,1,0))
        debug_im.SetOrigin(np.min(coords, axis=0))
        debug_im.SetDirection(imgVol.GetDirection())
        debug_im.SetSpacing((np.max(coords, axis=0)-np.min(coords, axis=0))/(np.array(GRID_SIZE)-1))
        sitk.WriteImage(debug_im, 'debug_patch.nii.gz')
        imgVol.SetSpacing(1./(np.array(IMAGE_SIZE)-1))
        sitk.WriteImage(imgVol, 'debug.nii.gz')
    if write:
        pickle.dump(info, open(os.path.join(os.path.dirname(__file__),datetime.now().strftime("%m_%d_%Y_%H_%M_%S")+"_pixel2mesh_aux_fit{}_{}_order{}.dat".format(num_pts, mode, order)),"wb"), protocol=2)

def make_dat_file_ffd_multi_level_grid_equal(num_pts, num_level,  template_processed, mode, order=0, image_fn=None, write=True):
    from scipy import sparse
    template, node_list, faces, bounds = template_processed
    coords = vtk_to_numpy(template.GetPoints().GetData())

    info = {'grid_coords': None, 'struct_node_ids':node_list, 'tmplt_faces': faces,
            'ffd_matrix_mesh': [None]*num_level, 'ffd_matrix_grid': [None]*(num_level-1),
            'support': [None]*num_level, 'grid_upsample': [None]*(num_level-1), 'ffd_matrix_image': [None]*(num_level)}
    info['tmplt_coords'] = coords
    MAX = 16
    prev_ctrl_grid = None
    for l in range(num_level):
        grid = make_grid(min(num_pts*(2**l), MAX), bounds)
        ctrl_pts = vtk_to_numpy(grid.GetPoints().GetData())
        if info['grid_coords'] is None:
            info['grid_coords'] = ctrl_pts
        #build adjacency matrix for the grid
        adj = np.zeros((len(ctrl_pts), len(ctrl_pts)))
        lines = vtk_to_numpy(grid.GetLines().GetData()).reshape(grid.GetNumberOfLines(), 3)
        lines = lines[:, 1:]
        for i in range(lines.shape[0]):
            adj[lines[i, 0], lines[i, 1]] = 1
            adj[lines[i, 1], lines[i, 0]] = 1
        adj_s = sparse.csr_matrix(adj)
        cheb = chebyshev_polynomials(adj_s,1)
        info['support'][l] = cheb
        if mode == 'ffd':
            info['ffd_matrix_mesh'][l] = ffd(ctrl_pts, coords)[-1]
            if l > 0:
                info['ffd_matrix_grid'][l-1] = ffd(prev_ctrl_grid, ctrl_pts, bounds)[-1]
                info['grid_upsample'][l-1] = transition_matrix_for_multi_level_grid(prev_ctrl_grid, ctrl_pts)
        elif mode=='bspline':
            info['ffd_matrix_mesh'][l] = sparse_to_tuple(construct_bspline_volume(ctrl_pts, coords, bounds))
            if l > 0:
                info['grid_upsample'][l-1] = transition_matrix_for_multi_level_grid(prev_ctrl_grid, ctrl_pts)
                info['ffd_matrix_grid'][l-1] = sparse_to_tuple(construct_bspline_volume(prev_ctrl_grid, ctrl_pts,bounds))
                print("debug: ", l-1, info['ffd_matrix_grid'][l-1])
        else:
            raise NotImplementedError
        if l < num_level - 1:
            prev_ctrl_grid = ctrl_pts
    if write:
        pickle.dump(info, open(os.path.join(os.path.dirname(__file__),datetime.now().strftime("%m_%d_%Y_%H_%M_%S")+"_pixel2mesh_aux_multi_level{}_fit{}_{}_order{}.dat".format(num_level, num_pts, mode, order)),"wb"), protocol=2)
def make_dat_file_ffd_multi_level_grid(num_pts, num_level,  template_processed, mode, order=0, image_fn=None, write=True):
    from scipy import sparse
    template, node_list, faces, bounds = template_processed
    coords = vtk_to_numpy(template.GetPoints().GetData())

    info = {'grid_coords': None, 'struct_node_ids':node_list, 'tmplt_faces': faces,
            'ffd_matrix_mesh': [None]*num_level, 'ffd_matrix_grid': [None]*num_level,
            'support': None, 'grid_downsample': [None]*(num_level-1), 'ffd_matrix_image': [None]*(num_level)}
    info['tmplt_coords'] = coords
    MAX = 16
    fine_grid = make_grid(min(num_pts*(2**(num_level-1)), MAX), bounds)
    fine_grid_pts = vtk_to_numpy(fine_grid.GetPoints().GetData())
    info['grid_coords'] = fine_grid_pts
    #build adjacency matrix for the grid
    adj = np.zeros((len(fine_grid_pts), len(fine_grid_pts)))
    lines = vtk_to_numpy(fine_grid.GetLines().GetData()).reshape(fine_grid.GetNumberOfLines(), 3)
    lines = lines[:, 1:]
    for i in range(lines.shape[0]):
        adj[lines[i, 0], lines[i, 1]] = 1
        adj[lines[i, 1], lines[i, 0]] = 1
    adj_s = sparse.csr_matrix(adj)
    cheb = chebyshev_polynomials(adj_s,1)
    info['support'] = cheb
    for l in range(num_level):
        grid = make_grid(min(num_pts*(2**l), MAX), bounds)
        ctrl_pts = vtk_to_numpy(grid.GetPoints().GetData())
        if mode == 'ffd':
            info['ffd_matrix_mesh'][l] = ffd(ctrl_pts, coords, bounds)[-1]
            info['ffd_matrix_grid'][l] = ffd(ctrl_pts, fine_grid_pts, bounds)[-1]
            if l < num_level-1:
                info['grid_downsample'][l] = transition_matrix_for_multi_level_grid(ctrl_pts, fine_grid_pts, True)
        elif mode=='bspline':
            info['ffd_matrix_mesh'][l] = sparse_to_tuple(construct_bspline_volume(ctrl_pts, coords, bounds))
            info['ffd_matrix_grid'][l] = sparse_to_tuple(construct_bspline_volume(ctrl_pts, fine_grid_pts, bounds))
            if l < num_level-1:
                grid_b= construct_bspline_volume(ctrl_pts, fine_grid_pts, bounds, order=order)
                info['grid_downsample'][l] = transition_matrix_for_multi_level_grid(ctrl_pts, fine_grid_pts, True)
        else:
            raise NotImplementedError
    if write:
        pickle.dump(info, open(os.path.join(os.path.dirname(__file__),datetime.now().strftime("%m_%d_%Y_%H_%M_%S")+"_pixel2mesh_aux_down_level{}_fit{}_{}_order{}.dat".format(num_level, num_pts, mode, order)),"wb"), protocol=2)

def make_dat_file_ffd_downsampled_grid(num_pts, num_level,  template_processed, mode, order=0, image_fn=None, option='equal', write=True):
    from scipy import sparse
    template, node_list, faces, bounds = template_processed
    print("BOUNDS: ", bounds)
    coords = vtk_to_numpy(template.GetPoints().GetData())
    info = {'sample_coords': None, 'struct_node_ids':node_list, 'tmplt_faces': faces, 'grid_coords': None, 
            'ffd_matrix_mesh': [None]*num_level,  
            'support': [None]*num_level, 'grid_downsample': [None]*num_level, 'ffd_matrix_image': [None]*num_level, 'grid_upsample':[None]*(num_level-1)}
    info['tmplt_coords'] = coords
    MAX = 16 
    if image_fn is not None:
        GRID_SIZE = 32
        IMAGE_SIZE = (128, 128, 128)
        image = sitk.ReadImage(image_fn)
        imgVol = resample_spacing(image, template_size=IMAGE_SIZE, order=1)[0]
        img_py = RescaleIntensity(sitk.GetArrayFromImage(imgVol).transpose(2,1,0), 'ct', [750, -750])
        print("Image_py", img_py, np.min(img_py), np.max(img_py))
        image_grid = make_grid(GRID_SIZE, bounds)
        info['image_data'] = get_image_patch(img_py, image_grid_coords)
        print("Image data: ", info['image_data'])

    sampling_grid = vtk.vtkPolyData()
    sampling_grid.DeepCopy(template)
    sample_pts = vtk_to_numpy(sampling_grid.GetPoints().GetData())
    info['sample_coords'] = sample_pts 
    for l in range(num_level):
        if l >0:
            prev_grid_pts = ctrl_pts
        if option=='equal':
            grid = make_grid(num_pts, bounds)
        else:
            grid = make_grid(min(num_pts*(2**l), MAX), bounds)
        ctrl_pts = vtk_to_numpy(grid.GetPoints().GetData())
        if info['grid_coords'] is None:
            info['grid_coords'] = ctrl_pts
        #build adjacency matrix for the grid
        adj = np.zeros((len(ctrl_pts), len(ctrl_pts)))
        lines = vtk_to_numpy(grid.GetLines().GetData()).reshape(grid.GetNumberOfLines(), 3)
        lines = lines[:, 1:]
        for i in range(lines.shape[0]):
            adj[lines[i, 0], lines[i, 1]] = 1
            adj[lines[i, 1], lines[i, 0]] = 1
        #print("Degree: ", np.sum(adj, axis=-1))
        #print(adj.shape)
        adj_s = sparse.csr_matrix(adj)
        cheb = chebyshev_polynomials(adj_s,1)
        info['support'][l] = cheb 
        if mode == 'ffd':
            info['ffd_matrix_mesh'][l] = ffd(ctrl_pts, coords, bounds)[-1]
        elif mode=='bspline':
            info['ffd_matrix_mesh'][l] = sparse_to_tuple(construct_bspline_volume(ctrl_pts, coords, bounds))
        else:
            raise NotImplementedError
        info['grid_downsample'][l] = transition_matrix_for_multi_level_grid_gaussion(sample_pts, ctrl_pts)
        if l >0:
            info['grid_upsample'][l-1] = transition_matrix_for_multi_level_grid(prev_grid_pts, ctrl_pts)
        # Debug: test down sampling matrix
        #debug = vtk.vtkPolyData()
        #debug.DeepCopy(sampling_grid)
        #sum_dist_vtk = numpy_to_vtk(info['grid_downsample'][l][0, :])
        #print("Point 0: ", ctrl_pts[0, :])
        #sum_dist_vtk.SetName('Down_sample_{}'.format(0))
        #debug.GetPointData().AddArray(sum_dist_vtk)
        #write_vtk_polydata(debug, os.path.join(os.path.dirname(__file__), 'down_sample_weights_level{}.vtp'.format(l)))
        # FFD for the image for now only on last block
        if image_fn is not None:
            if mode == 'ffd':
                info['ffd_matrix_image'][l] = ffd(ctrl_pts, image_grid_coords, bounds)[-1]
            elif mode=='bspline':
                mesh_b  = construct_bspline_volume(ctrl_pts, image_grid_coords,  bounds)
                info['ffd_matrix_image'][l] = sparse_to_tuple(mesh_b)
            else:
                raise NotImplementedError
            # debug
            debug_im = sitk.GetImageFromArray(info['image_data'].reshape(GRID_SIZE, GRID_SIZE, GRID_SIZE).transpose(2,1,0))
            debug_im.SetOrigin(np.min(coords, axis=0))
            debug_im.SetDirection(imgVol.GetDirection())
            debug_im.SetSpacing((np.max(coords, axis=0)-np.min(coords, axis=0))/(np.array(GRID_SIZE)-1))
            sitk.WriteImage(debug_im, 'debug_patch.nii.gz')
            imgVol.SetSpacing(1./(np.array(IMAGE_SIZE)-1))
            sitk.WriteImage(imgVol, 'debug.nii.gz')
    if write:
        pickle.dump(info, open(os.path.join(os.path.dirname(__file__), '../examples/mmwhs_test/dat_of_template_with_no_veins.dat'),"wb"), protocol=2)
        #pickle.dump(info, open(os.path.join(os.path.dirname(__file__),datetime.now().strftime("%m_%d_%Y_%H_%M_%S")+"_pixel2mesh_aux_down_level{}{}_fit{}_{}_order{}.dat".format(num_level, option, num_pts, mode, order)),"wb"), protocol=2)

    # debug
    #ctrl_pts = info['grid_coords']
    ##ctrl_pts[500, :] -= np.random.rand(3)
    #grid_deform = vtk.vtkPolyData()
    #grid_deform.DeepCopy(grid)
    #grid_deform.GetPoints().SetData(numpy_to_vtk(ctrl_pts))`
    #write_vtk_polydata(grid_deform, os.path.join(os.path.dirname(__file__), 'deformed_grid{}_debug.vtp'.format(num_pts)))
    #output = ffd(ctrl_pts, info['tmplt_coords'])
    #output_poly = vtk.vtkPolyData()
    #output_poly.DeepCopy(template)
    #output_poly.GetPoints().SetData(numpy_to_vtk(output))
    #write_vtk_polydata(output_poly, os.path.join(os.path.dirname(__file__), 'deformed_debug.vtp'))

def process_model_faces(fn):
    mesh = load_vtk_mesh(fn)
    model_face_ids = mesh.GetCellData().GetArray('ModelFaceID')
    py_array = vtk_to_numpy(model_face_ids)
    unique = np.unique(py_array)
    for index, i in enumerate(list(unique)):
        py_array[py_array==i] = index
    model_face_ids = numpy_to_vtk(py_array)
    model_face_ids.SetName('ModelFaceID_modified')
    mesh.GetCellData().AddArray(model_face_ids)
    write_vtk_polydata(mesh, fn)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_pts', type=int, help='Number of control points per dimension')
    parser.add_argument('--template', help='Path to the template heart model')
    parser.add_argument('--ref_template', help='Path to the reference template heart model')
    parser.add_argument('--template_im', default=None, help='Path to the template image that matches with the heart model')
    parser.add_argument('--target_node_num', type=int, help='Number of nodes desired')
    args = parser.parse_args()
    #process_model_faces(args.template)
    template = process_template(args.template, args.target_node_num, template_im_fn=args.template_im, ref_template_fn=args.ref_template)
    #test_ffd(grid, template[0], template[-1])
    #make_dat_file_ffd(args.num_pts, template, 'bspline', write=True)
    #make_dat_file_ffd_multi_level_grid(args.num_pts, 3, template, 'bspline', 3, args.template_im, write=True)
    #make_dat_file_ffd_multi_level_grid_equal(args.num_pts, 3, template, 'bspline', 3, args.template_im, write=True)
    make_dat_file_ffd_downsampled_grid(args.num_pts, 3, template, 'bspline', 3, None, option='double', write=True)

