import tensorflow as tf
import numpy as np
import astra
import nrrd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import os

distance_source_origin = 800 # [mm]
distance_origin_detector = 400 # [mm]
detector_pixel_size = 0.616 # [mm]
detector_rows = 480 # Vertical size of detector [pixels].
detector_cols = 620 # Horizontal size of detector [pixels].
num_of_projections = 360
num_of_sparse_projections = 13
angles = np.linspace(0, 2 * np.pi, num=num_of_projections, endpoint=False) # in radian
sparse_angles = np.linspace(0, 2 * np.pi, num=num_of_sparse_projections, endpoint=False) # in radian
sparse_indices = np.linspace(0, 360, num=num_of_sparse_projections, dtype=np.int, endpoint=False)

def _decode_validation_data(example_proto):
    feature = tf.io.parse_single_example(
        example_proto,
        features={
            'reco_tensor': tf.io.FixedLenFeature([], tf.string),
            'depth': tf.io.FixedLenFeature([], tf.int64),
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            'subject_file': tf.io.FixedLenFeature([], tf.string),
            'needle_file': tf.io.FixedLenFeature([], tf.string),
        })

    reco_tensor = tf.io.decode_raw(feature['reco_tensor'], tf.float32)
    depth = feature['depth']
    height = feature['height']
    width = feature['width']
    subject_file = feature['subject_file']
    needle_file = feature['needle_file']

    tensor = tf.reshape(reco_tensor, (depth, height, width, 3))
    tensor = tf.transpose(tensor, perm=(2, 1, 0, 3))

    return tensor, subject_file, needle_file

def _decode_projections(example_proto):
    feature = tf.io.parse_single_example(
        example_proto,
        features={
            'projections': tf.io.FixedLenFeature([], tf.string),
            'pixel_spacing_u': tf.io.FixedLenFeature([], tf.float32),
            'pixel_spacing_v': tf.io.FixedLenFeature([], tf.float32),
            'voxel_spacing_x': tf.io.FixedLenFeature([], tf.float32),
            'voxel_spacing_y': tf.io.FixedLenFeature([], tf.float32),
            'voxel_spacing_z': tf.io.FixedLenFeature([], tf.float32),
            'volume_depth': tf.io.FixedLenFeature([], tf.int64),
            'volume_height': tf.io.FixedLenFeature([], tf.int64),
            'volume_width': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            'height': tf.io.FixedLenFeature([], tf.int64),
            'angles': tf.io.FixedLenFeature([], tf.int64),
        })

    projections = tf.io.decode_raw(feature['projections'], tf.float32)
    # pixel_spacing_u = tf.io.decode_raw(feature['pixel_spacing_u'], tf.float32)
    # pixel_spacing_v = tf.io.decode_raw(feature['pixel_spacing_v'], tf.float32)
    width = feature['width']
    height = feature['height']
    angles = feature['angles']
    voxel_spacing = (
        feature['voxel_spacing_x'],
        feature['voxel_spacing_y'],
        feature['voxel_spacing_z'],
    )
    volume_shape = (
        feature['volume_depth'],
        feature['volume_height'],
        feature['volume_width'],
    )

    return tf.reshape(projections, (width, height, angles)), \
        voxel_spacing, volume_shape

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    prior_path = '/project/sghosh/dataset/mayoclinic/Head/high_dose_reconstructions'
    proj_path = '/project/sghosh/dataset/mayoclinic/Head/high_dose_projections/cone-beam/'
    needle_path = '/project/sghosh/dataset/mayoclinic/Head/high_dose_projections/needles/'
    recons_path = '/project/sghosh/dataset/mayoclinic/Head/test/'
    test_subjects = ["N216c", "N177c", "N270c", "N090c", "N160c", "N105c", "N176c"]
    needle_names = ['Needle2_Pos1_11','Needle2_Pos2_12','Needle2_Pos3_13']
    save_path ='/project/sghosh/experiments/predictions/13/sirt/'
    pred_path = os.path.join(save_path,'predictions')
    gt_path = os.path.join(save_path,'gt')
    os.mkdir(pred_path)
    os.mkdir(gt_path)
    for subject in test_subjects:
        for needle_name in needle_names:
            name = subject+'_'+needle_name
            prior = nrrd.read(os.path.join(prior_path, subject + '.nrrd'))[0]
            print(prior.shape)
            needle_dataset = tf.data.TFRecordDataset(os.path.join(needle_path, needle_name + '.tfrecord'),
                                                     num_parallel_reads=tf.data.experimental.AUTOTUNE)
            needle_dataset = needle_dataset.map(_decode_projections, num_parallel_calls=tf.data.experimental.AUTOTUNE)

            proj_dataset = tf.data.TFRecordDataset(os.path.join(proj_path, subject + '.tfrecord'),
                                                   num_parallel_reads=tf.data.experimental.AUTOTUNE)
            proj_dataset = proj_dataset.map(_decode_projections, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            for element in proj_dataset.as_numpy_iterator():
                sub_projections, _, _ = element

            for element in needle_dataset.as_numpy_iterator():
                needle_projections, _, _ = element

            # permute data to ASTRA convention
            projs = tf.cast(sub_projections, dtype=tf.float32)
            projs = np.transpose(projs, (1, 2, 0))
            projs = np.ascontiguousarray(projs)
            sparse_projs = projs[:, sparse_indices, :]
            sparse_projs = np.ascontiguousarray(sparse_projs)

            projs_with_needle = tf.cast(needle_projections, dtype=tf.float32)
            projs_with_needle = np.transpose(projs_with_needle, (1, 2, 0))
            projs_with_needle = projs + projs_with_needle
            projs_with_needle = np.ascontiguousarray(projs_with_needle)
            sparse_projs_with_needle = projs_with_needle[:, sparse_indices, :]
            sparse_projs_with_needle = np.ascontiguousarray(sparse_projs_with_needle)

            proj_n_geom = astra.create_proj_geom('cone', 1, 1,
                                                 detector_rows, detector_cols, angles, distance_source_origin,
                                                 distance_origin_detector)
            proj_n_id = astra.data3d.link('-sino', proj_n_geom, projs_with_needle)

            sp_proj_geom = astra.create_proj_geom('cone', 1, 1,
                                                  detector_rows, detector_cols, sparse_angles, distance_source_origin,
                                                  distance_origin_detector)
            sp_proj_id = astra.data3d.link('-sino', sp_proj_geom, sparse_projs_with_needle)

            proj_geom = astra.create_proj_geom('cone', 1, 1,
                                               detector_rows, detector_cols, angles, distance_source_origin,
                                               distance_origin_detector)
            proj_id = astra.data3d.link('-sino', proj_geom, projs)

            # get recons using FDK
            vol_geom = astra.creators.create_vol_geom(497, 451, 388)
            reconstruction_id = astra.data3d.create('-vol', vol_geom, data=0)

            alg_cfg = astra.astra_dict('FDK_CUDA')
            alg_cfg['ProjectionDataId'] = proj_id
            alg_cfg['ReconstructionDataId'] = reconstruction_id
            algorithm_id = astra.algorithm.create(alg_cfg)
            astra.algorithm.run(algorithm_id)
            reconstruction = astra.data3d.get(reconstruction_id)

            # SIRT
            vol_geom_sirt = astra.creators.create_vol_geom(497, 451, 388)
            # use the volume above as init for SIRT
            reconstruction_id_sirt = astra.data3d.create('-vol', vol_geom_sirt, data=0)
            cfg = astra.astra_dict('SIRT3D_CUDA')
            cfg['ReconstructionDataId'] = reconstruction_id_sirt
            cfg['ProjectionDataId'] = proj_n_id
            alg_sirt_id = astra.algorithm.create(cfg)
            # Run
            astra.algorithm.run(alg_sirt_id, 100)
            rec = astra.data3d.get(reconstruction_id_sirt)
            rec[rec < 0] = 0
            gt = rec / np.max(rec)
            print('gt')

            # SIRT
            vol_geom_sirt = astra.creators.create_vol_geom(497, 451, 388)
            # use the volume above as init for SIRT
            reconstruction_id_sirt = astra.data3d.create('-vol', vol_geom_sirt, data=0)
            cfg = astra.astra_dict('SIRT3D_CUDA')
            cfg['ReconstructionDataId'] = reconstruction_id_sirt
            cfg['ProjectionDataId'] = sp_proj_id
            alg_sirt_id = astra.algorithm.create(cfg)
            # Run
            astra.algorithm.run(alg_sirt_id, 100)
            rec = astra.data3d.get(reconstruction_id_sirt)
            rec[rec < 0] = 0
            noisy = rec / np.max(rec)
            print('noisy')
            np.save(os.path.join(pred_path, name+'.npy'), noisy)
            np.save(os.path.join(gt_path, name + '.npy'), gt)
            print(name)
