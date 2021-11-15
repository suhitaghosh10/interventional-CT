# def generate_datasets_dct(batch_size, buffer_size=1024):
#
#     #load training subjects
#     with open(JSON_PATH, 'r') as file_handle:
#         json_dict = json.load(file_handle)
#         train_subjects = json_dict['train_subjects']
#
#     # load con-beam projections of head
#     file_paths = [
#         os.path.join(TRAIN_CONEBEAM_PROJECTIONS_PATH, f'{tr}.tfrecord')
#         for tr in train_subjects
#     ]
#     volumes_dataset = tf.data.TFRecordDataset(file_paths)
#     vol_ds = volumes_dataset.map(_decode_vol_projections)  # ([w, h, 360], [3], [3])
#
#     # load needle projections
#     file_paths = [
#         os.path.join(NEEDLE_PROJECTIONS_PATH, filename)
#         for filename in os.listdir(NEEDLE_PROJECTIONS_PATH)
#         if filename.endswith('.tfrecord')
#     ]
#     needle_dataset = tf.data.TFRecordDataset(file_paths)
#     ndl_ds = needle_dataset.map(_decode_needle_projections)  # ([u, v, 360], [3], [3])
#    # ndl_ds = ndl_ds.map(_random_rotate)  # [u, v, 360]
#
#     # load prior helical scans
#     file_paths = [
#         os.path.join(TRAIN_HELICAL_PRIOR_PATH, f'{tr}.tfrecord')
#         for tr in train_subjects
#     ]
#     prior_ds = tf.data.TFRecordDataset(file_paths)
#     prior_ds = prior_ds.map(_decode_prior)
#
#     # training set
#     combined_ds = tf.data.Dataset.zip((vol_ds, ndl_ds))  # (([u, v, 360], [3], [3]), [u, v, 360])
#     combined_ds = combined_ds.map(_tensorize)
#     # generate the 3D reconstructions from cone-beam head and needle projections
#     combined_ds = combined_ds.map(
#         lambda x0, x1, x2, y: tf.numpy_function(func=_reconstruct_3D, inp=[x0, x1, x2, y], Tout=tf.float32),
#     )  # [w, h, d, 2]
#     tds = combined_ds.map(lambda x: tf.transpose(x, (2, 1, 0, 3)))  # [d, h, w, 2]
#     del combined_ds, needle_dataset, vol_ds, volumes_dataset
#
#     tds = tf.data.Dataset.zip((tds, prior_ds))  # ([d, h, w, 2], [d, h, w, 1])
#     tds = tds.map(lambda x, y: tf.concat([x, y], axis=3))  # [d, h, w, 3]
#     tds = tds.unbatch()  # [h, w, 3]
#     #tds = tds.map(_hu2normalized)
#     tds = tds.batch(batch_size)
#     tds = tds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
#
#
#
#     return tds
