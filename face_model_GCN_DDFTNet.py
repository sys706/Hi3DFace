import tensorflow as tf
import numpy as np
import tools.basis_utils as basis_op
import tools.uv_utils as uv_op
from tools.render_utils import Projector
from tools.const import OrthoCam
from net_model import *

def build_model_gcn_DDFTNet(images, basis3dmm=None, trainable=True, is_fine_model=False, is_medium_model=False):

    images224 = tf.image.resize_images(images, (224, 224))
    images224 = (images224 - 127.5) / 127.5

    # build coarse model
    para_shape, para_exp, para_tex, para_pose, para_illum = CoarseModel_Resnet(basis3dmm, images224, is_training=trainable)

    # get vertices (geometry and texture)
    ver_xyz = basis_op.get_geometry(basis3dmm, para_shape, para_exp)
    ver_rgb = basis_op.get_texture(basis3dmm, para_tex)
    coarse_tri = tf.constant(basis3dmm['tri'])
    coarse_ver_tri = tf.constant(basis3dmm['ver_tri'], tf.int32)

    if is_fine_model or is_medium_model:
        screen_size = 300
        images = tf.image.resize_images(images, (screen_size, screen_size))
    else:
        screen_size = 300

    # build transform matrix from pose para
    batch_size, imageW, imageH = images.get_shape().as_list()[0:3]
    trans = Projector.calc_trans_matrix(para_pose, screen_size)
    proj_xyz, ver_norm, ver_mask_contour, tri_norm = Projector.project(
            ver_xyz, trans, coarse_ver_tri, coarse_tri, imageW, imageH, OrthoCam, "projector")

    ver_mask_face = tf.expand_dims(
            tf.concat([basis3dmm['mask_face'].astype(np.float32)] * batch_size, axis=0, name='face_mask'),
            -1)
    ver_mask_wo_eye = tf.expand_dims(
            tf.concat([basis3dmm['mask_wo_eye'].astype(np.float32)] * batch_size, axis=0, name='wo_eye_mask'),
            -1)
    ver_mask_wo_nose = tf.expand_dims(
            tf.concat([basis3dmm['mask_wo_nose'].astype(np.float32)] * batch_size, axis=0, name='wo_nose_mask'),
            -1)
    ver_mask_wo_eyebrow = tf.expand_dims(
            tf.concat([basis3dmm['mask_wo_eyebrow'].astype(np.float32)] * batch_size, axis=0, name='wo_eyebrow_mask'),
            -1)
    ver_mask_attrs = tf.concat([ver_mask_face, ver_mask_wo_eye, ver_mask_wo_nose, ver_mask_wo_eyebrow], -1)

    # render images
    render_rgb, render_depth, render_normal, render_mask = Projector.sh_render(
            ver_xyz,
            ver_rgb / 255.0,
            ver_mask_attrs,
            coarse_ver_tri,
            coarse_tri,
            trans,
            para_illum,
            images / 255.0,
            OrthoCam,
            "projector"
            )
    tf.summary.image('render_normal', render_normal, max_outputs=3)

    render_rgb = render_rgb[:,:,:,:3] * 255.0

    render_mask_face = render_mask[:,:,:,0:1]
    render_mask_wo_eye = render_mask[:,:,:,1:2]
    render_mask_wo_nose = render_mask[:,:,:,2:3]
    render_mask_wo_eyebrow = render_mask[:,:,:,3:4]
    render_mask_photo = render_mask_face * render_mask_wo_eye
    render_rgb = render_rgb * render_mask_face + images * (1 - render_mask_face)

    coarse_results = {}
    coarse_results['para'] = {
            'shape': para_shape,
            'exp': para_exp,
            'tex': para_tex,
            'pose': para_pose,
            'illum': para_illum
            }
    coarse_results['ver'] = {
            'proj_xy': proj_xyz[:,:,:2],
            'xyz': ver_xyz,
            'rgb': ver_rgb,
            'normal': ver_norm,
            'face_mask': ver_mask_face,
            'wo_eye_mask': ver_mask_wo_eye,
            'wo_eyebrow_mask': ver_mask_wo_eyebrow,
            'wo_nose_mask': ver_mask_wo_nose,
            'contour_mask': ver_mask_contour
            }
    coarse_results['screen'] = {
            'rgb': render_rgb,
            'depth': render_depth,
            'wo_eye_mask': render_mask_wo_eye,
            'wo_eyebrow_mask': render_mask_wo_eyebrow,
            'wo_nose_mask': render_mask_wo_nose,
            'photo_mask': render_mask_photo
            }

    ret_results = {}
    ret_results['coarse'] = coarse_results

    if is_medium_model:

        # TODO: debug masks
        ver_mask_visible = tf.cast(tf.greater(ver_norm[:,:,2:3], 0.0), tf.float32) * ver_mask_face

        ver_mask_disp = ver_mask_wo_eye * ver_mask_wo_nose * ver_mask_wo_eyebrow

        # prepare uv indices for unwrapping
        ver_uv_index = basis_op.add_z_to_UV(basis3dmm)
        ver_uv_index = tf.tile(ver_uv_index[np.newaxis,...], [batch_size,1,1])

        uv_size = 512
        proj_coords = proj_xyz[:,:,:2]
        uv_inputs = uv_op.unwrap_screen_into_uv(images, proj_coords, coarse_tri, ver_uv_index, uv_size)  # (8, 512, 512, 3)
        uv_normal = uv_op.unwrap_screen_into_uv(render_normal, proj_coords, coarse_tri, ver_uv_index, uv_size)  # (8, 512, 512, 3)
        uv_rgb = uv_op.unwrap_screen_into_uv(render_rgb, proj_coords, coarse_tri, ver_uv_index, uv_size)  # (8, 512, 512, 3)

        uv_xyz = uv_op.convert_ver_attrs_into_uv(ver_xyz, ver_uv_index, coarse_tri, uv_size, uv_size)  # (8, 512, 512, 3)
        uv_rgb_tmp = uv_op.convert_ver_attrs_into_uv(ver_rgb, ver_uv_index, coarse_tri, uv_size, uv_size)  # (8, 512, 512, 3)
        uv_mask_face = uv_op.convert_ver_attrs_into_uv(ver_mask_face, ver_uv_index, coarse_tri, uv_size, uv_size)  # (8, 512, 512, 1)
        uv_mask_disp = uv_op.convert_ver_attrs_into_uv(ver_mask_disp, ver_uv_index, coarse_tri, uv_size, uv_size)  # (8, 512, 512, 1)
        uv_mask_visible = uv_op.convert_ver_attrs_into_uv(ver_mask_visible, ver_uv_index, coarse_tri, uv_size, uv_size)  # (8, 512, 512, 1)

        # debug visible mask
        tf.summary.image('uv_mask_visible', tf.cast(uv_mask_visible * 255, tf.uint8), max_outputs=3)
        tf.summary.image('render_depth', render_depth, max_outputs=3)

        uv_rgb = tf.clip_by_value(uv_rgb, 0, 255)
        uv_rgb_tmp = tf.clip_by_value(uv_rgb_tmp, 0, 255)
        uv_inputs = tf.clip_by_value(uv_inputs, 0, 255)
        uv_mask_face = tf.clip_by_value(uv_mask_face, 0, 1)
        uv_mask_disp = tf.clip_by_value(uv_mask_disp, 0, 1)
        uv_mask_visible = tf.clip_by_value(uv_mask_visible, 0, 1)

        tf.summary.image('uv_rgb_new', uv_rgb_tmp, max_outputs=3)

        # build depth displacement model
        fine_model = FineModel_DDFTNet()
        uv_disp_depth = fine_model.generator(uv_inputs, is_training=trainable)
        tf.summary.image('uv_disp_depth', uv_disp_depth, max_outputs=3)

        fine_topo = uv_op.TopoUV2Ver(uv_size, basis3dmm['uv_face_mask'], 'dense')
        new_ver_uv = tf.constant(fine_topo.ver_uv)
        fine_tri = tf.constant(fine_topo.triangles)
        fine_tri_tri = tf.constant(fine_topo.tri_tri, tf.int32)
        fine_ver_tri = tf.constant(fine_topo.ver_tri, tf.int32)

        ver_rgb_new = uv_op.remesh_uv_to_ver(uv_rgb_tmp, new_ver_uv)
        ver_xyz_new = uv_op.remesh_uv_to_ver(uv_xyz, new_ver_uv)
        ver_disp_new = uv_op.remesh_uv_to_ver(uv_disp_depth, new_ver_uv)
        ver_mask_face_new = uv_op.remesh_uv_to_ver(uv_mask_face, new_ver_uv)
        ver_norm_new = uv_op.remesh_uv_to_ver(uv_normal, new_ver_uv)
        ver_mask_disp_new = uv_op.remesh_uv_to_ver(uv_mask_disp, new_ver_uv)
        ver_mask_visible_new = uv_op.remesh_uv_to_ver(uv_mask_visible, new_ver_uv)

        ver_x, ver_y, ver_z = tf.split(ver_xyz_new, 3, axis=-1)
        norm_x, norm_y, norm_z = tf.split(ver_norm_new, 3, axis=-1)

        ver_xyz_fine = tf.concat([ver_x + ver_disp_new * norm_x * ver_mask_disp_new, ver_y + ver_disp_new * norm_y * ver_mask_disp_new, ver_z + ver_disp_new * norm_z * ver_mask_disp_new], axis=-1)

        proj_xyz_fine, ver_norm_fine, _, tri_norm_fine = Projector.project(
            ver_xyz_fine, trans, fine_ver_tri, fine_tri, imageW, imageH, OrthoCam, "projector_fine"
            )

        render_rgb_fine, render_depth_fine, render_normal_fine, render_mask_fine = Projector.sh_render(
            ver_xyz_fine,
            ver_rgb_new / 255.0,
            ver_mask_face_new,
            fine_ver_tri,
            fine_tri,
            trans,
            para_illum,
            images / 255.0,
            OrthoCam,
            "projector_fine"
        )

        render_rgb_fine = render_rgb_fine[:,:,:,:3] * 255.0

        render_rgb_fine_bg = render_rgb_fine * render_mask_face + images * (1 - render_mask_face)

        tf.summary.image('render_depth_fine', render_depth_fine, max_outputs=3)
        tf.summary.image('render_normal_fine', render_normal_fine, max_outputs=3)

        fine_results = {}

        fine_results['ver'] = {}
        fine_results['ver']['xyz'] = ver_xyz_fine
        fine_results['ver']['adj'] = tf.constant(fine_topo.ver_neighbors)
        fine_results['ver']['rgb_new'] = ver_rgb_new
        fine_results['ver']['normal'] = ver_norm_fine
        fine_results['ver']['coarse_normal'] = ver_norm_new
        fine_results['ver']['disp_depth'] = ver_disp_new
        fine_results['ver']['mask_face'] = ver_mask_face_new
        fine_results['ver']['mask_disp'] = ver_mask_disp_new

        fine_results['tri'] = {}
        fine_results['tri']['index'] = fine_tri
        fine_results['tri']['normal'] = tri_norm_fine
        fine_results['tri']['adj'] = fine_tri_tri

        fine_results['uv'] = {}
        fine_results['uv']['mask_face'] = uv_mask_face
        fine_results['uv']['mask_visible'] = uv_mask_visible
        fine_results['uv']['rgb'] = uv_rgb
        fine_results['uv']['input'] = uv_inputs
        fine_results['uv']['albedo_new'] = uv_rgb_tmp
        fine_results['uv']['disp_depth'] = uv_disp_depth
        fine_results['uv']['xyz_new'] = uv_xyz
        uv_x, uv_y, uv_z = tf.split(uv_xyz, 3, axis=-1)
        uv_z = uv_z + uv_disp_depth
        uv_xyz_fine = tf.concat([uv_x, uv_y, uv_z], axis=-1)
        fine_results['uv']['xyz'] = uv_xyz_fine
        tf.summary.image('uv_xyz', uv_xyz_fine, max_outputs=3)

        fine_results['screen'] = {}
        fine_results['screen']['rgb_fine'] = render_rgb_fine_bg
        fine_results['screen']['normal'] = render_normal_fine
        fine_results['screen']['inputs'] = images
        fine_results['screen']['depth'] = render_depth_fine
        fine_results['screen']['mask_face'] = render_mask_fine

        ret_results['medium'] = fine_results

    elif is_fine_model:

        # TODO: debug masks
        ver_mask_visible = tf.cast(tf.greater(ver_norm[:,:,2:3], 0.0), tf.float32) * ver_mask_face

        ver_mask_disp = ver_mask_wo_eye * ver_mask_wo_nose * ver_mask_wo_eyebrow

        # prepare uv indices for unwrapping
        ver_uv_index = basis_op.add_z_to_UV(basis3dmm)
        ver_uv_index = tf.tile(ver_uv_index[np.newaxis,...], [batch_size,1,1])

        uv_size = 512

        proj_coords = proj_xyz[:,:,:2]
        uv_inputs = uv_op.unwrap_screen_into_uv(images, proj_coords, coarse_tri, ver_uv_index, uv_size)  # (8, 512, 512, 3)
        uv_normal = uv_op.unwrap_screen_into_uv(render_normal, proj_coords, coarse_tri, ver_uv_index, uv_size)  # (8, 512, 512, 3)
        uv_rgb = uv_op.unwrap_screen_into_uv(render_rgb, proj_coords, coarse_tri, ver_uv_index, uv_size)  # (8, 512, 512, 3)

        uv_xyz = uv_op.convert_ver_attrs_into_uv(ver_xyz, ver_uv_index, coarse_tri, uv_size, uv_size)  # (8, 512, 512, 3)
        uv_rgb_tmp = uv_op.convert_ver_attrs_into_uv(ver_rgb, ver_uv_index, coarse_tri, uv_size, uv_size)  # (8, 512, 512, 3)
        uv_mask_face = uv_op.convert_ver_attrs_into_uv(ver_mask_face, ver_uv_index, coarse_tri, uv_size, uv_size)  # (8, 512, 512, 1)
        uv_mask_disp = uv_op.convert_ver_attrs_into_uv(ver_mask_disp, ver_uv_index, coarse_tri, uv_size, uv_size)  # (8, 512, 512, 1)
        uv_mask_visible = uv_op.convert_ver_attrs_into_uv(ver_mask_visible, ver_uv_index, coarse_tri, uv_size, uv_size)  # (8, 512, 512, 1)

        # debug visible mask
        tf.summary.image('uv_mask_visible', tf.cast(uv_mask_visible * 255, tf.uint8), max_outputs=3)
        tf.summary.image('render_depth', render_depth, max_outputs=3)

        uv_rgb = tf.clip_by_value(uv_rgb, 0, 255)
        uv_rgb_tmp = tf.clip_by_value(uv_rgb_tmp, 0, 255)
        uv_inputs = tf.clip_by_value(uv_inputs, 0, 255)
        uv_mask_face = tf.clip_by_value(uv_mask_face, 0, 1)
        uv_mask_disp = tf.clip_by_value(uv_mask_disp, 0, 1)
        uv_mask_visible = tf.clip_by_value(uv_mask_visible, 0, 1)

        tf.summary.image('uv_rgb_new', uv_rgb_tmp, max_outputs=3)

        # build depth displacement model
        fine_model = FineModel_DDFTNet()
        uv_disp_depth = fine_model.generator(uv_inputs, is_training=trainable)
        tf.summary.image('uv_disp_depth', uv_disp_depth, max_outputs=3)

        fine_topo = uv_op.TopoUV2Ver(uv_size, basis3dmm['uv_face_mask'], 'dense')
        new_ver_uv = tf.constant(fine_topo.ver_uv)
        fine_tri = tf.constant(fine_topo.triangles)
        fine_tri_tri = tf.constant(fine_topo.tri_tri, tf.int32)
        fine_ver_tri = tf.constant(fine_topo.ver_tri, tf.int32)

        ver_rgb_new = uv_op.remesh_uv_to_ver(uv_inputs, new_ver_uv)
        ver_xyz_new = uv_op.remesh_uv_to_ver(uv_xyz, new_ver_uv)
        ver_disp_new = uv_op.remesh_uv_to_ver(uv_disp_depth, new_ver_uv)
        ver_mask_face_new = uv_op.remesh_uv_to_ver(uv_mask_face, new_ver_uv)
        ver_norm_new = uv_op.remesh_uv_to_ver(uv_normal, new_ver_uv)
        ver_mask_disp_new = uv_op.remesh_uv_to_ver(uv_mask_disp, new_ver_uv)
        ver_mask_visible_new = uv_op.remesh_uv_to_ver(uv_mask_visible, new_ver_uv)

        ver_x, ver_y, ver_z = tf.split(ver_xyz_new, 3, axis=-1)
        norm_x, norm_y, norm_z = tf.split(ver_norm_new, 3, axis=-1)

        ver_xyz_fine = tf.concat([ver_x + ver_disp_new * norm_x * ver_mask_disp_new, ver_y + ver_disp_new * norm_y * ver_mask_disp_new, ver_z + ver_disp_new * norm_z * ver_mask_disp_new], axis=-1)

        proj_xyz_fine, ver_norm_fine, _, tri_norm_fine = Projector.project(
            ver_xyz_fine, trans, fine_ver_tri, fine_tri, imageW, imageH, OrthoCam, "projector_fine"
            )

        # generate refine ver_rgb
        laplacians = basis3dmm['laplacians_fine']  # Laplacian of the adjacencies matrix
        mesh_refine = MeshRefiner()
        ver_rgb_fine = mesh_refine.generator(ver_rgb_new, laplacians, is_training=trainable)
        ver_rgb_fine = (ver_rgb_fine + 1.0) / 2.0

        para_illum_fine = LightModel(images224, para_illum, is_training=trainable)  # refiner the illumination model

        render_rgb_fine, render_depth_fine, render_normal_fine, render_mask_fine = Projector.sh_render(
            ver_xyz_fine,
            ver_rgb_fine,
            ver_mask_face_new,
            fine_ver_tri,
            fine_tri,
            trans,
            para_illum_fine,
            images / 255.0,
            OrthoCam,
            "projector_fine"
        )

        render_rgb_fine = render_rgb_fine[:,:,:,:3] * 255.0
        render_rgb_fine_bg = render_rgb_fine * render_mask_face + images * (1 - render_mask_face)

        tf.summary.image('render_depth_fine', render_depth_fine, max_outputs=3)
        tf.summary.image('render_normal_fine', render_normal_fine, max_outputs=3)

        fine_results = {}

        fine_results['ver'] = {}
        fine_results['ver']['xyz'] = ver_xyz_fine
        fine_results['ver']['adj'] = tf.constant(fine_topo.ver_neighbors)
        fine_results['ver']['rgb_new'] = ver_rgb_new
        fine_results['ver']['rgb_fine'] = ver_rgb_fine
        fine_results['ver']['normal'] = ver_norm_fine
        fine_results['ver']['coarse_normal'] = ver_norm_new
        fine_results['ver']['disp_depth'] = ver_disp_new
        fine_results['ver']['mask_face'] = ver_mask_face_new
        fine_results['ver']['mask_disp'] = ver_mask_disp_new

        fine_results['tri'] = {}
        fine_results['tri']['index'] = fine_tri
        fine_results['tri']['normal'] = tri_norm_fine
        fine_results['tri']['adj'] = fine_tri_tri

        fine_results['uv'] = {}
        fine_results['uv']['mask_face'] = uv_mask_face
        fine_results['uv']['mask_visible'] = uv_mask_visible
        fine_results['uv']['rgb'] = uv_rgb
        fine_results['uv']['input'] = uv_inputs
        fine_results['uv']['albedo_new'] = uv_rgb_tmp
        fine_results['uv']['disp_depth'] = uv_disp_depth
        fine_results['uv']['xyz_new'] = uv_xyz
        uv_x, uv_y, uv_z = tf.split(uv_xyz, 3, axis=-1)
        uv_z = uv_z + uv_disp_depth
        uv_xyz_fine = tf.concat([uv_x, uv_y, uv_z], axis=-1)
        fine_results['uv']['xyz'] = uv_xyz_fine
        tf.summary.image('uv_xyz', uv_xyz_fine, max_outputs=3)

        fine_results['screen'] = {}
        fine_results['screen']['rgb_fine'] = render_rgb_fine_bg
        fine_results['screen']['normal'] = render_normal_fine
        fine_results['screen']['inputs'] = images
        fine_results['screen']['depth'] = render_depth_fine
        fine_results['screen']['mask_face'] = render_mask_fine
        ret_results['fine'] = fine_results

    return ret_results
