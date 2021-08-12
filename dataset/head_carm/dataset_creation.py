from utility.common_imports import *
from utility.utils import augment_prior, insert_needle
from dataset.head_carm.constants import *

CARMHEAD_2D_TFRECORDS_TRAIN = 'carmhead.tfrecords.train'
CARMHEAD_2D_TFRECORDS_VAL = 'carmhead.tfrecords.val'
CARMHEAD_2D_TFRECORDS_TEST = 'carmhead.tfrecords.test'

TRAIN = 'train'
VAL = 'val'
TEST = 'test'

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def generate_perceptual_dataset(data_path, batch_size=1, buffer_size=1024):
    file_paths = []

    for folder, subs, files in os.walk(os.path.join(data_path, TRAIN)):
        for filename in files:
            file_paths.append(os.path.abspath(os.path.join(folder, filename)))
    # np.random.random(file_paths)
    train_dataset = tf.data.TFRecordDataset(file_paths, num_parallel_reads=6)
    val_dataset = tf.data.TFRecordDataset(os.path.join(data_path , VAL, CARMHEAD_2D_TFRECORDS_VAL))
    test_dataset = tf.data.TFRecordDataset(os.path.join(data_path, TEST, CARMHEAD_2D_TFRECORDS_TEST))

    tds = train_dataset.map(_read_and_decode, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    tds = tds.shuffle(buffer_size=buffer_size).map(augment_prior).batch(batch_size=batch_size)
    tds = tds.repeat()
    tds = tds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    vds = val_dataset.map(_read_and_decode, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    vds = vds.map(insert_needle).batch(batch_size=batch_size)
    vds = vds.repeat()
    vds = vds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    teds = test_dataset.map(_read_and_decode, num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(buffer_size=buffer_size)
    teds = teds.repeat()
    teds = teds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return tds, vds, teds

def _read_and_decode(example_proto):

    feature = tf.io.parse_single_example(
        example_proto,
        features={
            'name': tf.io.FixedLenFeature([], tf.string),
            'img': tf.io.FixedLenFeature([], tf.string),
            'gt': tf.io.FixedLenFeature([], tf.string)
        })

    start = (IMG_DIM_ORIG_2D[0] - IMG_DIM_INP_2D[0]) // 2

    image = tf.io.decode_raw(feature['img'], tf.float16)
    annotation = tf.io.decode_raw(feature['gt'], tf.float16)
    image= tf.reshape(image, (IMG_DIM_ORIG_2D[0], IMG_DIM_ORIG_2D[1], 1))
    annotation = tf.reshape(annotation, (IMG_DIM_ORIG_2D[0], IMG_DIM_ORIG_2D[1], 1))
    #return [384, 384, 1]
    return annotation[start:start+IMG_DIM_INP_2D[0], start:start+IMG_DIM_INP_2D[1],:], \
           image[start:start+IMG_DIM_INP_2D[0], start:start+IMG_DIM_INP_2D[1],:]
