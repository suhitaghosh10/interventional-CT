import tensorflow as tf
import numpy as np
import astra
import nrrd
import matplotlib.pyplot as plt
import os


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'


    proj_path = '/project/sghosh/dataset/mayoclinic/Head/high_dose_projections/cone-beam/'
    needle_path = '/project/sghosh/dataset/mayoclinic/Head/high_dose_projections/needles/'
    recons_path = '/project/sghosh/dataset/mayoclinic/Head/test/'
    subjects = []
    for subject in os.listdir(recons_path):
        subjects.append(subject)

    num_subjects = len(subjects)
    print(subject.split('_'))
    # subject = test_subjects[0]
    #
    # recons_dataset = tf.data.TFRecordDataset(os.path.join(recons_path, subject + '_' + needle_name + '.tfrecord'),
    #                                          num_parallel_reads=tf.data.experimental.AUTOTUNE)
    # recons_dataset = recons_dataset.map(_decode_validation_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    #
    # needle_dataset = tf.data.TFRecordDataset(os.path.join(needle_path, needle_name + '.tfrecord'),
    #                                          num_parallel_reads=tf.data.experimental.AUTOTUNE)
    # needle_dataset = needle_dataset.map(_decode_projections, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    #
    # proj_dataset = tf.data.TFRecordDataset(os.path.join(proj_path, subject + '.tfrecord'),
    #                                        num_parallel_reads=tf.data.experimental.AUTOTUNE)
    # proj_dataset = proj_dataset.map(_decode_projections, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    #
    # for element in recons_dataset.as_numpy_iterator():
    #     recon_tensor, _, _ = element
    #
    # for element in proj_dataset.as_numpy_iterator():
    #     sub_projections, _, _ = element
    #
    # for element in needle_dataset.as_numpy_iterator():
    #     needle_projections, _, _ = element
    #
    # sparse_recon_w_ndl = recon_tensor[:, :, :, 0]
    # prior = recon_tensor[:, :, :, 1]
    # gt = recon_tensor[:, :, :, 2]
    # shape = sparse_recon_w_ndl.shape
    #
    # distance_source_origin = 800  # [mm]
    # distance_origin_detector = 400  # [mm]
    # detector_pixel_size = 0.616  # [mm]
    # detector_rows = 480  # Vertical size of detector [pixels].
    # detector_cols = 620  # Horizontal size of detector [pixels].
    # num_of_projections = 360
    # num_of_sparse_projections = 18
    # angles = np.linspace(0, 2 * np.pi, num=num_of_projections, endpoint=False)  # in radian
    # sparse_angles = np.linspace(0, 2 * np.pi, num=num_of_sparse_projections, endpoint=False)  # in radian
    # sparse_indices = np.linspace(0, 360, num=num_of_sparse_projections, dtype=np.int, endpoint=False)
    # print(sparse_angles)
    # print(sparse_indices)
    #
    # # permute data to ASTRA convention
    # projs = tf.cast(sub_projections, dtype=tf.float32)
    # projs = np.transpose(projs, (1, 2, 0))
    # projs = np.ascontiguousarray(projs)
    # sparse_projs = projs[:, sparse_indices, :]
    # sparse_projs = np.ascontiguousarray(sparse_projs)
    # projs.shape, sparse_projs.shape
    #
    # projs_with_needle = tf.cast(needle_projections, dtype=tf.float32)
    # projs_with_needle = np.transpose(projs_with_needle, (1, 2, 0))
    # projs_with_needle = projs + projs_with_needle
    # projs_with_needle = np.ascontiguousarray(projs_with_needle)
    # sparse_projs_with_needle = projs_with_needle[:, sparse_indices, :]
    # sparse_projs_with_needle = np.ascontiguousarray(sparse_projs_with_needle)
    # projs_with_needle.shape, sparse_projs_with_needle.shape
    #
    # proj_n_geom = astra.create_proj_geom('cone', 1, 1,
    #                                      detector_rows, detector_cols, angles, distance_source_origin,
    #                                      distance_origin_detector)
    # proj_n_id = astra.data3d.link('-sino', proj_n_geom, projs_with_needle)
    #
    # sp_proj_geom = astra.create_proj_geom('cone', 1, 1,
    #                                       detector_rows, detector_cols, sparse_angles, distance_source_origin,
    #                                       distance_origin_detector)
    # sp_proj_id = astra.data3d.link('-sino', sp_proj_geom, sparse_projs_with_needle)
    #
    # proj_geom = astra.create_proj_geom('cone', 1, 1,
    #                                    detector_rows, detector_cols, angles, distance_source_origin,
    #                                    distance_origin_detector)
    # proj_id = astra.data3d.link('-sino', proj_geom, projs)
    #
    # vol_geom = astra.creators.create_vol_geom(497, 451, 388)
    # reconstruction_id = astra.data3d.create('-vol', vol_geom, data=0)
    #
    # alg_cfg = astra.astra_dict('FDK_CUDA')
    # alg_cfg['ProjectionDataId'] = proj_n_id
    # alg_cfg['ReconstructionDataId'] = reconstruction_id
    # algorithm_id = astra.algorithm.create(alg_cfg)
    # astra.algorithm.run(algorithm_id)
    # sp_reconstruction = astra.data3d.get(reconstruction_id)
    #
    # vol_geom = astra.creators.create_vol_geom(497, 451, 388)
    # reconstruction_id = astra.data3d.create('-vol', vol_geom, data=0)
    #
    # alg_cfg = astra.astra_dict('FDK_CUDA')
    # alg_cfg['ProjectionDataId'] = proj_id
    # alg_cfg['ReconstructionDataId'] = reconstruction_id
    # algorithm_id = astra.algorithm.create(alg_cfg)
    # astra.algorithm.run(algorithm_id)
    # reconstruction = astra.data3d.get(reconstruction_id)
    # reconstruction[reconstruction < 0] = 0
    # reconstruction /= np.max(reconstruction)
    # reconstruction = np.round(reconstruction * 255).astype(np.uint8)
    # reconstruction.shape
    #
    # # SIRT
    # # Configure geometry
    # # reconstruction_id_sirt = astra.data3d.create('-vol', reconstruction_id, data=reconstruction_id)
    # # Configure algorithm
    # vol_geom_sirt = astra.creators.create_vol_geom(497, 451, 388)
    # reconstruction_id_sirt = astra.data3d.create('-vol', vol_geom_sirt, data=0)
    # cfg = astra.astra_dict('SIRT3D_CUDA')
    # cfg['ReconstructionDataId'] = reconstruction_id_sirt
    # cfg['ProjectionDataId'] = sp_proj_id
    # alg_sirt_id = astra.algorithm.create(cfg)
    # # Run
    # astra.algorithm.run(alg_sirt_id, 100)
    # rec = astra.data3d.get(reconstruction_id_sirt)
    # rec[rec < 0] = 0
    # rec /= np.max(rec)
    # rec = np.round(rec * 255).astype(np.uint8)
