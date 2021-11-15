#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 18:23:46 2018

DEN and DICOM IO manipulation

@author: Vojtech Kulvait
@author: Enno Schieferecke

Edited by :
Hana Haseljić
Soumick Chatterjee
Suhita Ghosh

"""
import numpy as np
import skimage
import os

# import cv2

__author__ = "Vojtech Kulvait, Hana Haseljić, Soumick Chatterjee"
__copyright__ = "Copyright 2019, Vojtech Kulvait Hana Haseljić Soumick Chatterjee"
__credits__ = ["Hana Haseljić"]

__license__ = "GPL"
__version__ = "1.0.0"
__email__ = "hana.haseljic@ovgu.de"
__status__ = "Finished"


# Get frame from DEN file
# Can get a subframe, where row_from is the first row index
# row_to is not included
# col_from is the first col index
# col_to is not included

def read_den_volume(fileName, block, type):
    header = readHeader(fileName)
    vol = np.zeros((header['rows'], header['cols'], header['zdim']))
    #print(header['zdim'])read_den_volume
    for i in range(header['zdim']):
        #print(i)
        vol[:, :, i] = getFrame(fileName, i, block=block, type=type)
    return vol


def getFrame(fileName, i, row_from=None, row_to=None, col_from=None, col_to=None, block=4, type=np.dtype('<f4')):
    # i is the slice index
    header = np.fromfile(fileName, np.dtype('<i2'), 3)
    rows = np.uint32(header[0])
    columns = np.uint32(header[1])
    if row_from is None:
        row_from = 0
    if row_to is None:
        row_to = rows
    if col_from is None:
        col_from = 0
    if col_to is None:
        col_to = columns
    f = open(fileName, "rb")
    f.seek(6 + rows * columns * block * i, os.SEEK_SET)
    data = np.fromfile(f, type, rows * columns)
    newdata = data.reshape((rows, columns))
    newdata = newdata[row_from:row_to, col_from:col_to]
    f.close()
    return (newdata)


def storeNdarrayAsDEN(fileName, dataFrame, force=False, block=4,type=np.dtype('<f4')):
    if not force and os.path.exists(fileName):
        raise IOError('File already exists, no data written')
    if not isinstance(dataFrame, np.ndarray):
        raise TypeError('Object dataFrame has to be of type numpy.array')
    if len(dataFrame.shape) == 1:
        print('Dimension = 1, expected >= 2')
        return False
    elif len(dataFrame.shape) == 2:
        dataFrame = np.expand_dims(dataFrame, axis=2)
    elif len(dataFrame.shape) > 3:
        raise ValueError(
            'Dimension of dataFrame should be 2 or 3 but is %d.' % len(dataFrame.shape))
    shape = dataFrame.shape  # Now len is for sure 3
    writeEmptyDEN(fileName, dimx=shape[0], dimy=shape[1], dimz=shape[2],
                  force=force)  # No effect
    header = np.fromfile(fileName, np.dtype('<i2'), 3)
    rows = np.uint32(header[0])
    columns = np.uint32(header[1])
    f = open(fileName, "wb")
    header.tofile(f)
    i = dataFrame.shape[-1]
    for frame in range(i):
        newdata = np.array(dataFrame[:, :, frame], dtype=type)
        newdata = newdata.reshape((rows * columns,))
        # put header in front of image data
        f.seek(6 + rows * columns * frame * block, os.SEEK_SET)
        newdata.tofile(f)
    f.close()
    return True


def writeFrame(fileName, k, data, force=False):
    if not force and os.path.exists(fileName):
        raise IOError('File %s already exists, no header written' % fileName)
    shape = data.shape
    if len(shape) != 2:
        raise ValueError('Dimension of data should be 2 %d.' % len(shape))
    header = np.fromfile(fileName, np.dtype('<i2'), 3)
    rows = np.uint32(header[0])
    columns = np.uint32(header[1])
    if shape[0] != rows or shape[1] != columns:
        raise ValueError(
            'There is dimension mismatch between frame (%d, %d) and expected (rows, cols) = (%d, %d) according to header.' %
            (rows, columns, shape[0], shape[1]))
    f = open(fileName, "ab")
    data = np.array(data, np.dtype('<f4'))
    data = data.reshape((rows * columns))
    # f.seek(6 + rows * columns * k * 4, os.SEEK_SET)
    data.tofile(f)
    f.close()


def writeEmptyDEN(fileName, dimx, dimy, dimz, force=False):
    if not force and os.path.exists(fileName):
        raise IOError('File %s already exists, no header written' % fileName)
    outfile = open(fileName, "w")
    header = np.array([dimx, dimy, dimz])
    header = np.array(header, dtype='<i2')
    header.tofile(outfile)
    # fileSize = dimx * dimy * dimz * 4 + 6
    # outfile.seek(fileSize - 1)
    # outfile.write('\0')
    outfile.close()


def readHeader(file):
    header = np.fromfile(file, np.dtype('<i2'), 3)
    par = {}
    par["rows"] = np.uint32(header[0])
    par["cols"] = np.uint32(header[1])
    par["zdim"] = np.uint32(header[2])
    return (par)


# Trim frame to the specified dimensions
# Can get a subframe, where row_from is the first row index
# row_to is not included
# col_from is the first col index
# col_to is not included


def trimFrame(frame, row_from, row_to, col_from, col_to):
    newdata = frame[row_from:row_to, col_from:col_to]
    return (newdata)


def compareDEN(a_path, b_path):
    a_header = readHeader(a_path)
    b_header = readHeader(b_path)
    if not np.array_equal(a_header, b_header):
        exit()

    num_of_frames = np.uint32(a_header["zdim"])
    mse = np.zeros(num_of_frames)
    ssim = np.empty(num_of_frames)
    # num_of_frames = 2
    writeEmptyDEN('ssim.den', np.uint32(a_header["rows"]), np.uint32(a_header["cols"]), num_of_frames, True)

    i = 0
    while (i < num_of_frames):
        a = getFrame(a_path, i, row_from=None, row_to=None, col_from=None, col_to=None)
        b = getFrame(b_path, i + 1, row_from=None, row_to=None, col_from=None, col_to=None)

        mse[i] = skimage.measure.compare_nrmse(a, b, 'EUCLIDEAN')
        ssim[i], smap = skimage.measure.compare_ssim(a, b, win_size=None, gradient=False, data_range=None,
                                                     multichannel=False, gaussian_weights=False, full=True)

        writeFrame('ssim.den', i, smap, force=True)

        i += 1

    return (mse, ssim)

if __name__ == '__main__':
    #npy = np.load('/home/suhita/memorial/dataset/LDCT_Preprocessed/gt/CHEST/C009/1-002.dcm.pkl', allow_pickle=True)
   # storeNdarrayAsDEN('/home/suhita/memorial/dataset/LDCT_Preprocessed/sad.den', npy, force=True)
   #  root_path='Z:/'
   #  folder_path = root_path+'MEMoRIAL_M1.p-3/dataset/carm_ct_from_shiras/'
   #  folders = os.listdir(folder_path)
   #  angles = np.linspace(start=0, stop=496, num=16, endpoint=False, dtype=np.int)
   #
   #  for folder in folders:
   #      print(folder)
   #      proj_path = os.path.join(folder_path, folder, 'proj.den')
   #      pmat_path = os.path.join(folder_path, folder, 'pmat.den')
   #      if os.path.exists(os.path.join(folder_path, folder, 'proj_15.den')) or os.path.exists(os.path.join(folder_path, folder, 'pmat_15.den')):
   #          continue
   #      elif os.path.exists(proj_path) and os.path.exists(pmat_path):
   #          proj = read_den_volume(proj_path, block=4, type=np.dtype('<f4'))
   #          temp = proj[:,:,angles]
   #          storeNdarrayAsDEN(os.path.join(folder_path, folder, 'proj_15.den'), temp, block=4,type=np.dtype('<f4'), force=True)
   #
   #          pmat = read_den_volume(pmat_path, block=8, type=np.double)
   #          #print(pmat.shape)
   #          temp = pmat[:, :, angles]
   #          storeNdarrayAsDEN(os.path.join(folder_path, folder, 'pmat_15.den'), temp, block=8, type=np.double, force=True)
   #      else:
   #          print(folder, 'doesnt exist!')

    #vol = read_den_volume('/scratch/sghosh/proj_15.den', block=4, type=np.dtype('<f4'))
    vol = read_den_volume('/data/suhita/memorial/dataset/carm_head/p37/proj.den', block=4, type=np.dtype('<f4'))
    print(vol.shape)
    #np.save('/scratch/sghosh/proj_15.npy', vol)
    #print(vol.shape)



