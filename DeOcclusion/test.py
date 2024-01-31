import random
from PIL import Image
import gc
import os
import cv2
import glob
import socket
import tensorflow as tf
from inpaint_model import *
from mask_online import *
import argparse

def infer(batch_data,mask,reuse=False):
        batch_gt=batch_data/127.5-1.
        batch_incomplete=batch_gt*mask
        image_p1, image_p2=RW_generator(batch_incomplete,mask,reuse=reuse)
        image_c2=batch_gt*mask+ image_p2*(1.-mask)
        image_c2 = (image_c2 + 1.) * 127.5
        return image_c2

if __name__=='__main__':
        parser = argparse.ArgumentParser(description='training code')
        parser.add_argument('--output',type=str ,default=" " ,help='test_data_path')
        parser.add_argument('--test_data_path',type=str ,default=" " ,help='test_data_path')
        parser.add_argument('--mask_path',type=str ,default=" " ,help='mask_path')
        parser.add_argument('--model_path',type=str ,default=" " ,help='model_path')
        parser.add_argument('--width',type=int ,default=256 ,help='images width')
        parser.add_argument('--height',type=int ,default=256 ,help='images height')
        args = parser.parse_args()

        if os.path.exists(args.output) is False:
            os.makedirs(args.output)
        
        images=tf.placeholder(tf.float32,[1,args.height,args.width,3],name = 'image')
        mask=tf.placeholder(tf.float32,[1,args.height,args.width,1],name='mask')
        sess = tf.Session()        
        inpainting_result=infer(images,mask)
        saver_pre=tf.train.Saver()
        init_op = tf.group(tf.initialize_all_variables(),tf.initialize_local_variables()) 
        sess.run(init_op)
        saver_pre.restore(sess,args.model_path)

        test_mask = cv2.resize(cv2.imread(args.mask_path),(args.height,args.width))
        test_mask = test_mask[:,:,0:1]
        test_mask  = 1 - test_mask
        test_image = cv2.imread(args.test_data_path)[...,::-1]
        test_image = cv2.resize(test_image, (args.height, args.width))
        test_mask = np.expand_dims(test_mask,0)
        test_image = np.expand_dims(test_image,0)
        img_out=sess.run(inpainting_result,feed_dict={mask:test_mask,images:test_image})
        name = args.test_data_path.split('/')[-1].split('.')[0]

        cv2.imwrite(os.path.join(args.output, name + '_DeOccluded.png'), img_out[0][...,::-1])


