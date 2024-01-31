""" basis related operations """
import tensorflow as tf
import glob
import skimage.io
import numpy as np
import scipy.io
import scipy
import imageio
import cv2
import json
import os
import scipy.sparse as sp
import tools.uv_utils as uv_op

def get_geometry(basis3dmm, para_shape, para_exp):
    """ compute the geometry according to the 3DMM parameters.
    para_shape: shape parameter (199d)
    para_exp: expression parameter (29d)
    """
    shape_inc = tf.matmul(para_shape, basis3dmm['bases_shape'])
    exp_inc = tf.matmul(para_exp, basis3dmm['bases_exp'])
    geo = basis3dmm['mu_shape'] + basis3dmm['mu_exp'] + shape_inc + exp_inc
    n_vtx = basis3dmm['bases_shape'].shape[1] // 3
    return tf.reshape(geo,[-1,n_vtx,3])

def get_texture(basis3dmm, para_tex):
    """ compute the texture according to texture parameter.
    para_tex: texture parameter (199d)
    """
    tex_inc = tf.matmul(para_tex,basis3dmm['bases_tex'])
    texture = tf.clip_by_value(basis3dmm['mu_tex'] + tex_inc,0,255)
    n_vtx = basis3dmm['bases_shape'].shape[1] // 3
    return tf.reshape(texture, [-1,n_vtx,3])

def add_z_to_UV(basis3dmm):
    uv = basis3dmm['ver_uv_ind']
    # add random values to prevent vertices from collapsing into the same pixels
    uv = uv + np.random.uniform(size=[uv.shape[0],2],low=0.,high=0.00001)
    uv = uv.astype(np.float32)
    z = np.reshape(basis3dmm['mu_shape'], [-1,3])[:,2:3]
    norm_z = (z - np.amin(z)) / (np.amax(z) - np.amin(z))
    uvz = np.concatenate([uv,-norm_z], axis=1)
    return uvz

def load_3dmm_basis_gcn(basis_path,
        ver_uv_ind_path,
        uv_face_mask_path,
        ver_wo_eyebrow_mask_path=None,
        ver_wo_nose_mask_path=None):
    basis3dmm = scipy.io.loadmat(basis_path)

    basis3dmm['bases_shape'] = np.transpose(basis3dmm['bases_shape'] * basis3dmm['sigma_shape']).astype(np.float32)
    basis3dmm['bases_exp'] = np.transpose(basis3dmm['bases_exp'] * basis3dmm['sigma_exp']).astype(np.float32)
    basis3dmm['bases_tex'] = np.transpose(basis3dmm['bases_tex'] * basis3dmm['sigma_tex']).astype(np.float32)
    basis3dmm['mu_shape'] = np.transpose(basis3dmm['mu_shape']).astype(np.float32)
    basis3dmm['mu_exp'] = np.transpose(basis3dmm['mu_exp']).astype(np.float32)
    basis3dmm['mu_tex'] = np.transpose(basis3dmm['mu_tex']).astype(np.float32)
    basis3dmm['tri'] = basis3dmm['tri'].astype(np.int32)

    # get vertex triangle relation shape
    vertex_tri_set = set()
    vertex_vertex_set = set()
    for i, tri in enumerate(basis3dmm['tri']):
        v1 = tri[0]
        v2 = tri[1]
        v3 = tri[2]
        vertex_tri_set.add((v1,i))
        vertex_tri_set.add((v2,i))
        vertex_tri_set.add((v1,i))
        vertex_tri_set.add((v3,i))
        vertex_tri_set.add((v2,i))
        vertex_tri_set.add((v3,i))

        vertex_vertex_set.add((v1, v2))
        vertex_vertex_set.add((v2, v1))
        vertex_vertex_set.add((v1, v3))
        vertex_vertex_set.add((v3, v1))
        vertex_vertex_set.add((v2, v3))
        vertex_vertex_set.add((v3, v2))
    vertex_tri_set = np.array(list(vertex_tri_set), np.int32)
    vertex_vertex_set = np.array(list(vertex_vertex_set), np.int32)
    basis3dmm['ver_tri'] = vertex_tri_set
    basis3dmm['ver_adj'] = vertex_vertex_set

    # crop bases
    basis3dmm['bases_shape'] = basis3dmm['bases_shape'][:80,:]
    basis3dmm['bases_tex'] = basis3dmm['bases_tex'][:80,:]

    ver_uv_ind = np.load(ver_uv_ind_path)['uv_ind']
    basis3dmm['ver_uv_ind'] = (ver_uv_ind / 512.0).astype(np.float32)

    uv_face_mask = cv2.imread(uv_face_mask_path)
    if len(uv_face_mask.shape) == 3:
        uv_face_mask = uv_face_mask[:,:,0] / 255.0
    uv_face_mask = uv_face_mask.astype(np.float32)
    basis3dmm['uv_face_mask'] = uv_face_mask

    # get the neighboring relationship of triangles
    edge_to_triangles = {}
    for idx, tri in enumerate(basis3dmm['tri']):
        v1 = tri[0]
        v2 = tri[1]
        v3 = tri[2]
        try:
            edge_to_triangles[(v1,v2)].append(idx)
        except Exception:
            edge_to_triangles[(v1,v2)] = [idx]

        try:
            edge_to_triangles[(v2,v1)].append(idx)
        except Exception:
            edge_to_triangles[(v2,v1)] = [idx]

        try:
            edge_to_triangles[(v1,v3)].append(idx)
        except Exception:
            edge_to_triangles[(v1,v3)] = [idx]

        try:
            edge_to_triangles[(v3,v1)].append(idx)
        except Exception:
            edge_to_triangles[(v3,v1)] = [idx]

        try:
            edge_to_triangles[(v2,v3)].append(idx)
        except Exception:
            edge_to_triangles[(v2,v3)] = [idx]

        try:
            edge_to_triangles[(v3,v2)].append(idx)
        except Exception:
            edge_to_triangles[(v3,v2)] = [idx]

    tri_pairs = []
    for key in edge_to_triangles:
        relations = edge_to_triangles[key]
        for item_a in relations:
            for item_b in relations:
                if item_a < item_b:
                    tri_pairs.append((item_a, item_b))
    tri_pairs = set(tri_pairs)
    tri_pairs = np.array(list(tri_pairs), np.int32)
    basis3dmm['tri_pairs'] = tri_pairs

    # get adjacencies and laplacians for coarse
    n_vers = int(basis3dmm['bases_shape'].shape[-1]/3)
    adjacencies = get_vert_connectivity(n_vers, basis3dmm['tri'])
    adjacencies = adjacencies.astype(np.float32)

    laplacians = laplacian(adjacencies, normalized=True)
    pool_size = adjacencies.shape[0]
    basis3dmm['adjacencies'] = adjacencies
    basis3dmm['laplacians'] = laplacians
    basis3dmm['pool_size'] = pool_size

    # get adjacencies and laplacians for refiner
    uv_size = 512
    fine_topo = uv_op.TopoUV2Ver(uv_size, uv_face_mask, 'dense')
    n_vers_fine = np.shape(fine_topo.ver_uv)[0]
    fine_tri = fine_topo.triangles

    adjacencies_fine = get_vert_connectivity(n_vers_fine, fine_tri)
    adjacencies_fine = adjacencies_fine.astype(np.float32)

    laplacians_fine = laplacian(adjacencies_fine, normalized=True)
    pool_size_fine = adjacencies_fine.shape[0]
    basis3dmm['adjacencies_fine'] = adjacencies_fine
    basis3dmm['laplacians_fine'] = laplacians_fine
    basis3dmm['pool_size_fine'] = pool_size_fine

    if ver_wo_eyebrow_mask_path is not None:
        ver_wo_eyebrow_mask = np.load(ver_wo_eyebrow_mask_path)
        ver_wo_eyebrow_mask = ver_wo_eyebrow_mask.astype(np.float32)
        ver_wo_eyebrow_mask = np.reshape(ver_wo_eyebrow_mask, [-1, 1])

        for i in range(10):
            ver_wo_eyebrow_mask = laplace_smoothing(ver_wo_eyebrow_mask, basis3dmm['ver_adj'])
        ver_wo_eyebrow_mask = np.squeeze(ver_wo_eyebrow_mask)
        basis3dmm['mask_wo_eyebrow'] = ver_wo_eyebrow_mask[np.newaxis,...]

    if ver_wo_nose_mask_path is not None:
        ver_wo_nose_mask = np.load(ver_wo_nose_mask_path)
        ver_wo_nose_mask = ver_wo_nose_mask.astype(np.float32)
        ver_wo_nose_mask = np.reshape(ver_wo_nose_mask, [-1, 1])

        for i in range(10):
            ver_wo_nose_mask = laplace_smoothing(ver_wo_nose_mask, basis3dmm['ver_adj'])
        ver_wo_nose_mask = np.squeeze(ver_wo_nose_mask)
        basis3dmm['mask_wo_nose'] = ver_wo_nose_mask[np.newaxis,...]

    return basis3dmm

def get_vert_connectivity(n_verts, mesh_f):
  """Returns a sparse matrix (of size #verts x #verts) where each nonzero
    element indicates a neighborhood relation. For example, if there is a
    nonzero element in position (15,12), that means vertex 15 is connected
    by an edge to vertex 12."""

  vpv = sp.csc_matrix((n_verts, n_verts))

  # for each column in the faces...
  for i in range(3):
    IS = mesh_f[:, i]
    JS = mesh_f[:, (i + 1) % 3]
    data = np.ones(len(IS))
    # ij = np.vstack((row(IS.flatten()), row(JS.flatten())))
    ij = np.vstack((IS.reshape((1, -1)), JS.reshape(1, -1)))
    mtx = sp.csc_matrix((data, ij), shape=vpv.shape)
    vpv = vpv + mtx + mtx.T

  return vpv

def laplacian(W, normalized=True):
  """Return the Laplacian of the weigth matrix."""

  # Degree matrix.
  d = W.sum(axis=0)

  # Laplacian matrix.
  if not normalized:
    D = scipy.sparse.diags(d.A.squeeze(), 0)
    L = D - W
  else:
    d += np.spacing(np.array(0, W.dtype))
    d = 1 / np.sqrt(d)
    D = scipy.sparse.diags(d.A.squeeze(), 0)
    I = scipy.sparse.identity(d.size, dtype=W.dtype)
    L = I - D * W * D

  assert isinstance(L, scipy.sparse.csr.csr_matrix)
  return L

def laplace_smoothing(ver, ver_adj):
    # ver: numpy array, [N, x]
    # ver_adj: [M, 2], (ver_id, ver_adj_id), vertex index starting from 0
    N = np.amax(ver_adj)
    assert(len(ver.shape) == 2)
    ver_attrs = np.zeros_like(ver)
    ver_counts = np.zeros([ver.shape[0], 1], np.float32)
    for (v_id, v_adj_id) in ver_adj:
        ver_attrs[v_id,:] = ver_attrs[v_id,:] + ver[v_adj_id,:]
        ver_counts[v_id,:] += 1
    ver_attrs = ver_attrs / (ver_counts + 1e-8)
    return ver_attrs



