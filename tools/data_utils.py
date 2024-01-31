import numpy as np
import tensorflow as tf
import os


class DataSet_Name(object):

    def __init__(self):
        """
        This dataset only contains image, name.
        image:
        name:
        """
        pass

    @staticmethod
    def parse(img_path, name):

        # input image
        x = tf.read_file(img_path)
        img = tf.image.decode_png(x, channels=3)
        img = tf.cast(img, tf.float32)
        image = tf.image.resize_images(img[:, :, :3], [300, 300])
        return image, name

    @staticmethod
    def load(args):
        """
        Load (image, name).

        :param:
            data_dir: string, test datasets directory
            batch_size: int, batch size
        """
        image_names = os.listdir(args.data_dir)
        image_names.sort()

        img_path = [os.path.join(args.data_dir, name) for name in image_names]

        dataset = tf.data.Dataset.from_tensor_slices((img_path, image_names))

        dataset = dataset.shuffle(buffer_size=len(image_names))

        #batch process
        # dataset = dataset.prefetch(buffer_size=args.batch_size*10)
        dataset = dataset.prefetch(buffer_size=args.batch_size)
        dataset = dataset.repeat(1)
        dataset = dataset.shuffle(buffer_size=512)

        dataset = dataset.apply(tf.contrib.data.map_and_batch(DataSet_Name.parse, batch_size=args.batch_size))
        iterator = dataset.make_one_shot_iterator()
        image, name = iterator.get_next()

        # reshape to get batch size
        screen_size = 300
        image = tf.reshape(image,[args.batch_size, screen_size, screen_size, 3])
        name = tf.reshape(name, [args.batch_size, 1])

        return image, name
