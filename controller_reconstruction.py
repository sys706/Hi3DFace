import tensorflow as tf
import os
import cv2
import glob
import numpy as np
from PIL import Image
import face_model_GCN_DDFTNet
from tools.basis_utils import load_3dmm_basis_gcn
from tools.data_utils import DataSet_Name
from tools.ply import write_ply
from tools.misc import MaskCreator


def reconstruction(args):
    # data loader
    image_load, name_load = DataSet_Name.load(args)

    basis3dmm = load_3dmm_basis_gcn(args.bfm_path, args.ver_uv_index, args.uv_face_mask_path, './resources/wo_eyebrow_mask.npy', './resources/wo_nose_mask.npy')

    print('building model')
    ret_results = face_model_GCN_DDFTNet.build_model_gcn_DDFTNet(image_load, basis3dmm, trainable=False, is_fine_model=args.is_fine_model, is_medium_model=args.is_medium_model)

    # trainable variables
    train_vars = tf.trainable_variables()
    global_vars = tf.global_variables()
    bn_moving_vars = [g for g in global_vars if 'moving_mean' in g.name]
    bn_moving_vars += [g for g in global_vars if 'moving_variance' in g.name]
    train_vars += bn_moving_vars

    coarse_vars = []
    medium_vars = []
    fine_vars = []
    if args.is_fine_model:
        for var in global_vars:
            if var.name.startswith('resnet_v1_50') or var.name.startswith('para_shape') or var.name.startswith('para_exp') \
                    or var.name.startswith('para_tex') or var.name.startswith('para_pose') or var.name.startswith('para_illum'):
                coarse_vars.append(var)
            elif var.name.startswith('DepthModel'):
                medium_vars.append(var)
            elif var.name.startswith('FineModel'):
                fine_vars.append(var)

    elif args.is_medium_model:
        for var in global_vars:
            if var.name.startswith('resnet_v1_50') or var.name.startswith('para_shape') or var.name.startswith('para_exp') \
                    or var.name.startswith('para_tex') or var.name.startswith('para_pose') or var.name.startswith('para_illum'):
                coarse_vars.append(var)
            elif var.name.startswith('DepthModel'):
                medium_vars.append(var)

    else:
        for var in global_vars:
            if var.name.startswith('resnet_v1_50') or var.name.startswith('para_shape') or var.name.startswith('para_exp') \
                    or var.name.startswith('para_tex') or var.name.startswith('para_pose') or var.name.startswith('para_illum'):
                coarse_vars.append(var)

    saver_coarse = None
    saver_medium = None
    saver_fine = None
    if args.is_fine_model:
        saver_coarse = tf.train.Saver(coarse_vars)
        saver_medium = tf.train.Saver(medium_vars)
        saver_fine = tf.train.Saver(fine_vars)
    elif args.is_medium_model:
        saver_coarse = tf.train.Saver(coarse_vars)
        saver_medium = tf.train.Saver(medium_vars)
    else:
        saver_coarse = tf.train.Saver(coarse_vars)
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    with tf.Session() as sess:

        sess.run(init_op)

        if args.is_fine_model:
            saver_coarse.restore(sess, args.load_coarse_ckpt)
            saver_medium.restore(sess, args.load_medium_ckpt)
            saver_fine.restore(sess, args.load_fine_ckpt)

        elif args.is_medium_model:
            saver_coarse.restore(sess, args.load_coarse_ckpt)
            saver_medium.restore(sess, args.load_medium_ckpt)

        else:
            saver_coarse.restore(sess, args.load_coarse_ckpt)

        if os.path.exists(args.output_dir) is False:
            os.makedirs(args.output_dir)

        for i in range(len(os.listdir(args.data_dir))):

            if args.is_fine_model:
                print("rendering ", i + 1, " images")
                outputs, input_images, img_names = sess.run([ret_results, image_load, name_load])
                name = img_names[0, 0].decode().split('.')[0]
                render_fine = outputs['fine']['screen']['rgb_fine'][0]
                cv2.imwrite(os.path.join(args.output_dir, '%s_fine_rendering.png' % name), render_fine[:, :, ::-1])
            elif args.is_medium_model:
                print("rendering ", i + 1, " images")
                outputs, input_images, img_names = sess.run([ret_results, image_load, name_load])
                name = img_names[0, 0].decode().split('.')[0]
                render_medium = outputs['medium']['screen']['rgb_fine'][0]
                cv2.imwrite(os.path.join(args.output_dir, '%s_medium_rendering.png' % name), render_medium[:, :, ::-1])
            else:
                print("rendering  ", i + 1, " images")
                outputs, input_images, img_names = sess.run([ret_results, image_load, name_load])
                image = input_images[0]
                name = img_names[0, 0].decode().split('.')[0]
                render_coarse = outputs['coarse']['screen']['rgb'][0]

                cv2.imwrite(os.path.join(args.output_dir, '%s_input.png' % name), image[:, :, ::-1])
                cv2.imwrite(os.path.join(args.output_dir, '%s_coarse_rendering.png' % name), render_coarse[:, :, ::-1])
