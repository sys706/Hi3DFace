import tensorflow as tf
import numpy as np
import scipy
from tensorflow.contrib.slim.nets import resnet_v1
slim = tf.contrib.slim

MAX = 1e5


def fft_filter(image, sigma):
    filtered_images = []
    for i in range(image.shape[0]):
        # Compute the 2D FFT of the image
        image_fft = tf.signal.fft2d(tf.cast(image[i, :, :, 0], dtype=tf.complex64))

        # Shift the zero-frequency component to the center of the spectrum
        image_fft_shifted = tf.signal.fftshift(image_fft)

        kernel = np.zeros((image.shape[2], image.shape[1]))
        kernel[image.shape[2] // 2 - sigma:image.shape[2] // 2 + sigma,
        image.shape[1] // 2 - sigma:image.shape[1] // 2 + sigma] = 1

        # Apply the Fourier Transform based filter to the shifted image
        filtered_image_fft_shifted = image_fft_shifted * tf.cast(kernel, dtype=tf.complex64)

        # Shift the zero-frequency component back to the corner of the spectrum
        filtered_image_fft = tf.signal.ifftshift(filtered_image_fft_shifted)

        # Compute the inverse FFT of the filtered image
        filtered_image = tf.cast(tf.abs(tf.signal.ifft2d(filtered_image_fft)), dtype=tf.float32)
        filtered_image = tf.expand_dims(filtered_image, -1)
        filtered_images.append(filtered_image)
    filtered_images = tf.cast(filtered_images, dtype=tf.float32)
    return filtered_images

def fft_filter_two(image, sigma):
    bach_images = []

    kernel = np.zeros((image.shape[2], image.shape[1]))
    kernel[image.shape[2] // 2 - sigma:image.shape[2] // 2 + sigma, image.shape[1] // 2 - sigma:image.shape[1] // 2 + sigma] = 1

    for i in range(image.shape[0]):
        channel_images = None
        for j in range(image.shape[3]):
            # Compute the 2D FFT of the image
            image_fft = tf.signal.fft2d(tf.cast(image[i, :, :, j], dtype=tf.complex64))

            # Shift the zero-frequency component to the center of the spectrum
            image_fft_shifted = tf.signal.fftshift(image_fft)

            # Apply the Fourier Transform based filter to the shifted image
            filtered_image_fft_shifted = image_fft_shifted * tf.cast(kernel, dtype=tf.complex64)

            # Shift the zero-frequency component back to the corner of the spectrum
            filtered_image_fft = tf.signal.ifftshift(filtered_image_fft_shifted)

            # Compute the inverse FFT of the filtered image
            filtered_image = tf.cast(tf.abs(tf.signal.ifft2d(filtered_image_fft)), dtype=tf.float32)
            filtered_image = tf.expand_dims(filtered_image, -1)
            if channel_images is None:
                channel_images = filtered_image
            else:
                channel_images = tf.concat([channel_images, filtered_image], axis=-1)

        bach_images.append(channel_images)
    bach_images = tf.cast(bach_images, dtype=tf.float32)
    return bach_images

def rescale_L(L, lmaxs=2):
  """Rescale the Laplacian eigenvalues in [-1,1]."""
  M, M = L.shape
  I = scipy.sparse.identity(M, format='csr', dtype=L.dtype)
  L /= lmaxs / 2
  L -= I
  return L

def chebyshev5(inputs, L, Fout, K, is_training):
    # self.InterX = x
    N, M, Fin = inputs.get_shape()
    N, M, Fin = int(N), int(M), int(Fin)
    # Rescale Laplacian and store as a TF sparse tensor. Copy to not modify the shared L.
    L = scipy.sparse.csr_matrix(L)
    L = rescale_L(L, 2)
    L = L.tocoo()
    indices = np.column_stack((L.row, L.col))
    L = tf.SparseTensor(indices, L.data, L.shape)
    L = tf.sparse_reorder(L)
    # Transform to Chebyshev basis
    x0 = tf.transpose(inputs, perm=[1, 2, 0])
    x0 = tf.reshape(x0, [M, Fin * N])
    x = tf.expand_dims(x0, 0)

    def concat(x, x_):
      x_ = tf.expand_dims(x_, 0)
      return tf.concat([x, x_], axis=0)

    if K > 1:
      x1 = tf.sparse_tensor_dense_matmul(L, x0)
      x = concat(x, x1)
    for _ in range(2, K):
      x2 = 2 * tf.sparse_tensor_dense_matmul(L, x1) - x0
      x = concat(x, x2)
      x0, x1 = x1, x2
    x = tf.reshape(x, [K, M, Fin, N])
    x = tf.transpose(x, perm=[3, 1, 2, 0])
    x = tf.reshape(x, [N * M, Fin * K])

    x = tf.layers.dense(x, Fout, activation=None, use_bias=False, kernel_initializer=tf.initializers.he_normal(), trainable=is_training)

    return tf.reshape(x, [N, M, Fout])


def CoarseModel_Resnet(basis3dmm, inputs, is_training=True):
    # inputs: [Batchsize, H, W, C], [-1, 1]
    inputs = tf.cast(inputs, tf.float32)
    # standard ResNet50 backbone (without the last classfication FC layer)
    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        net, end_points = resnet_v1.resnet_v1_50(inputs, is_training = is_training, reuse = tf.AUTO_REUSE)

    para_shape = slim.conv2d(net, basis3dmm['bases_shape'].shape[0],
                             [1, 1],
                             activation_fn=None,
                             normalizer_fn=None,
                             weights_initializer=tf.initializers.he_normal(),
                             trainable=is_training,
                             scope='para_shape')

    para_exp = slim.conv2d(net, basis3dmm['bases_exp'].shape[0],
                           [1, 1],
                           activation_fn=None,
                           normalizer_fn=None,
                           weights_initializer=tf.initializers.he_normal(),
                           trainable=is_training,
                           scope='para_exp')

    para_tex = slim.conv2d(net, basis3dmm['bases_tex'].shape[0],
                           [1, 1],
                           activation_fn=None,
                           normalizer_fn=None,
                           weights_initializer=tf.initializers.he_normal(),
                           trainable=is_training,
                           scope='para_tex')

    para_pose = slim.conv2d(net, 6,
                            [1, 1],
                            activation_fn=None,
                            normalizer_fn=None,
                            weights_initializer=tf.initializers.he_normal(),
                            trainable=is_training,
                            scope='para_pose')

    para_illum = slim.conv2d(net, 27,
                             [1, 1],
                             activation_fn=None,
                             normalizer_fn=None,
                             weights_initializer=tf.initializers.he_normal(),
                             trainable=is_training,
                             scope='para_illum')

    para_shape = tf.squeeze(para_shape, [1, 2], name='para_shape/squeezed')
    para_tex = tf.squeeze(para_tex, [1, 2], name='para_tex/squeezed')
    para_exp = tf.squeeze(para_exp, [1, 2], name='para_exp/squeezed')
    para_pose = tf.squeeze(para_pose, [1, 2], name='para_pose/squeezed')
    para_illum = tf.squeeze(para_illum, [1, 2], name='para_illum/squeezed')

    return para_shape, para_exp, para_tex, para_pose, para_illum


# displacement depth model
class FineModel_DDFTNet(object):
    def __init__(self):
        pass

    def generator(self, input_uv, is_training=True):
        """ auto-encoder for uv maps.
        :param:
            :input_uv: unwrap input uv maps.
            :render_coarse_uv: unwrap render coarse uv maps
            :mask_uv: indicating the visible region in uv masks
        :return:
            :disp_z: displacement in z direction.
        """

        inputs = input_uv / 255.0
        #################################

        with tf.variable_scope('DepthModel') as scope:
            # TODO: add high-pass filter

            conv0 = tf.layers.conv2d(inputs, 8, [3,3], 1, padding='SAME',trainable=is_training) # 512 -> 512
            conv0_act = tf.nn.relu(conv0)

            conv1 = tf.layers.conv2d(conv0_act, 16, [3,3], 2, padding='SAME',trainable=is_training) # 512 -> 256
            conv1_act = tf.nn.relu(conv1)

            conv2 = tf.layers.conv2d(conv1_act, 32, [3,3], 2, padding='SAME',trainable=is_training) # 256 -> 128
            conv2_act = tf.nn.relu(conv2)

            conv3 = tf.layers.conv2d(conv2_act, 64, [3,3], 2, padding='SAME',trainable=is_training) # 128 -> 64
            conv3_act = tf.nn.relu(conv3)

            conv4 = tf.layers.conv2d(conv3_act, 128, [3,3], 2, padding='SAME',trainable=is_training) # 64 -> 32
            conv4_act = tf.nn.relu(conv4)

            conv5 = tf.layers.conv2d(conv4_act, 256, [3,3], 2, padding='SAME',trainable=is_training) # 32 -> 16
            conv5_act = tf.nn.relu(conv5)

            conv6 = tf.layers.conv2d(conv5_act, 512, [3,3], 2, padding='SAME',trainable=is_training) # 16 -> 8
            conv6_act = tf.nn.relu(conv6)

            conv7 = tf.layers.conv2d(conv6_act, 512, [3,3], 2, padding='SAME',trainable=is_training) # 8 -> 4
            conv7_act = tf.nn.relu(conv7)

            deconv7 = tf.layers.conv2d_transpose(conv7_act, 512, [3,3], 2, padding='SAME',trainable=is_training) # 4 -> 8
            deconv7 = tf.nn.relu(deconv7) + conv6_act

            deconv6 = tf.layers.conv2d_transpose(tf.concat([deconv7, conv6_act], -1), 256, [3,3], 2, padding='SAME',trainable=is_training) # 8 -> 16
            deconv6 = tf.nn.relu(deconv6)

            deconv5 = tf.layers.conv2d_transpose(tf.concat([deconv6, conv5_act], -1), 128, [3,3], 2, padding='SAME',trainable=is_training) # 16 -> 32
            deconv5 = tf.nn.relu(deconv5)

            deconv4 = tf.layers.conv2d_transpose(tf.concat([deconv5, conv4_act], -1), 64, [3,3], 2, padding='SAME',trainable=is_training) # 32 -> 64
            deconv4 = tf.nn.relu(deconv4)

            deconv3 = tf.layers.conv2d_transpose(tf.concat([deconv4, conv3_act], -1), 32, [3,3], 2, padding='SAME',trainable=is_training) # 64 -> 128
            deconv3 = tf.nn.relu(deconv3)

            deconv2 = tf.layers.conv2d_transpose(tf.concat([deconv3, conv2_act], -1), 16, [3,3], 2, padding='SAME',trainable=is_training) # 128 -> 256
            deconv2 = tf.nn.relu(deconv2)

            deconv1 = tf.layers.conv2d_transpose(tf.concat([deconv2, conv1_act], -1), 8, [3,3], 2, padding='SAME',trainable=is_training) # 256 -> 512
            deconv1 = tf.nn.relu(deconv1)

            deconv0 = tf.layers.conv2d(fft_filter_two(deconv1, 70), 1, [3,3], 1, padding='SAME',trainable=is_training) # 256 -> 512
            depth_disp_uv = tf.nn.sigmoid(deconv0)
            depth_disp_uv = fft_filter(depth_disp_uv, 30)

        return depth_disp_uv

class MeshRefiner(object):
    def __int__(self):
        pass
    def generator(self, ver_rgb, laplacians, is_training=True):
        """ refiner GCN for ver_rgb.
        :param:
            :ver_rgb: vertices rgb.
            :laplacians: Laplacian of the adjacencies matrix.
        :return:
            :ver_rgb_fine: vertices rgb.
        """

        ver_rgb = ver_rgb / 127.5 -1.0  # the values [-1, 1]

        with tf.variable_scope('FineModel', reuse=tf.AUTO_REUSE) as scope:

            x0 = chebyshev5(ver_rgb, laplacians, 16, 6, is_training)
            x0 = tf.nn.relu(x0)
            x = chebyshev5(x0, laplacians, 32, 6, is_training)
            x = tf.nn.relu(x)
            x = chebyshev5(x, laplacians, 32, 6, is_training)
            x = tf.nn.relu(x)
            x = chebyshev5(x, laplacians, 16, 6, is_training)
            x = tf.nn.relu(x)
            ver_rgb_fine = chebyshev5(x, laplacians, 3, 6, is_training)
            ver_rgb_fine = tf.nn.tanh(ver_rgb_fine)

        return ver_rgb_fine

# illumination model for fine model
def LightModel(inputs, para_illum, is_training=True):
    # inputs: [Batchsize, H, W, C], [-1, 1]
    inputs = tf.cast(inputs, tf.float32)
    # standard ResNet50 backbone (without the last classfication FC layer)
    with tf.variable_scope('FineModel', reuse=tf.AUTO_REUSE):
        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            net, end_points = resnet_v1.resnet_v1_50(inputs, is_training=is_training, reuse=tf.AUTO_REUSE)
        # Modified FC layer with 128 channels for details coefficients
        net = tf.squeeze(net, [1, 2], name='net/squeezed')

        light_fc1 = tf.layers.dense(net,
                                    128,
                                    use_bias=False,
                                    kernel_initializer=tf.initializers.he_normal(),
                                    trainable=is_training,
                                    name='light_fc1')

        light_fc2 = tf.layers.dense(tf.concat([light_fc1, para_illum], -1),
                                    27,
                                    use_bias=False,
                                    kernel_initializer=tf.initializers.he_normal(),
                                    trainable=is_training,
                                    name='light_fc2')
    return light_fc2