import os
import sys

import numpy as np
from numpy.testing import assert_equal, assert_array_less

import skvideo
import skvideo.datasets
import skvideo.io

if sys.version_info < (2, 7):
    import unittest2 as unittest
else:
    import unittest


@unittest.skipIf(not skvideo._HAS_AVCONV, "LibAV required for this test.")
def test_LibAVReader_aboveversion9():
    # skip if libav not installed or of the proper version
    if not skvideo._HAS_AVCONV:
        return 0
    if np.int(skvideo._LIBAV_MAJOR_VERSION) < 9:
        return 0


    T = 0
    M = 0
    N = 0
    C = 0
    accumulation = 0
    with skvideo.io.LibAVReader(skvideo.datasets.bigbuckbunny(), verbosity=0) as reader:
        for frame in reader.nextFrame():
            M, N, C = frame.shape
            accumulation += np.sum(frame)
            T += 1

    # check the dimensions of the video
    assert_equal(T, 132)
    assert_equal(M, 720)
    assert_equal(N, 1280)
    assert_equal(C, 3)

    # check the numbers
    assert_equal(accumulation / (T * M * N * C), 109.28332841215979)


@unittest.skipIf(not skvideo._HAS_AVCONV, "LibAV required for this test.")
def test_LibAVReader_aboveversion9_fps():
    # skip if libav not installed or of the proper version
    if not skvideo._HAS_AVCONV:
        return 0
    if np.int(skvideo._LIBAV_MAJOR_VERSION) < 9:
        return 0

    T = 0
    M = 0
    N = 0
    C = 0
    accumulation = 0
    with skvideo.io.LibAVReader(skvideo.datasets.bigbuckbunny(), outputdict={"-r": "10"}, verbosity=0) as reader:
        for frame in reader.nextFrame():
            M, N, C = frame.shape
            accumulation += np.sum(frame)
            T += 1

    # check the dimensions of the video
    assert_equal(T, 53)
    assert_equal(M, 720)
    assert_equal(N, 1280)
    assert_equal(C, 3)

    # check the numbers
    assert_equal(accumulation / (T * M * N * C), 109.2392282699489)

@unittest.skipIf(not skvideo._HAS_AVCONV, "LibAV required for this test.")
def test_LibAVReader_aboveversion9_16bits():

    T = 0
    M = 0
    N = 0
    C = 0
    accumulation = 0

    with skvideo.io.LibAVReader(skvideo.datasets.bigbuckbunny(), outputdict={'-pix_fmt':'rgb48le',"-r":"5"}, verbosity=0) as reader16, \
         skvideo.io.LibAVReader(skvideo.datasets.bigbuckbunny(), outputdict={'-pix_fmt':'rgb24',"-r":"5"}, verbosity=0) as reader8:
        for frame8, frame16 in zip(reader8.nextFrame(), reader16.nextFrame()):
            # testing with the measure module may be a better idea but would add a dependency

            # check that there is no more than a 3/256th defference between the 8bit and 16 bit decoded image
            assert(np.max(np.abs(frame8.astype('int32') - (frame16//256).astype('int32'))) < 4)
            # check that the mean difference is less than 1
            assert(np.mean(np.abs(frame8.astype('float32') - (frame16//256).astype('float32'))) < 1.0)
            M, N, C = frame8.shape
            accumulation += np.sum(frame16//256)
            T += 1

    # check the dimensions of the video
    assert_equal(T, 27)
    assert_equal(M, 720)
    assert_equal(N, 1280)
    assert_equal(C, 3)

    # check the numbers : there's probably a better way to do this
    assert_equal(accumulation / (T * M * N * C), 108.7741166597008)

@unittest.skipIf(not skvideo._HAS_AVCONV, "LibAV required for this test.")
def test_LibAVWriter_aboveversion9():
    # skip if libav not installed or of the proper version
    if not skvideo._HAS_AVCONV:
        return 0
    if np.int(skvideo._LIBAV_MAJOR_VERSION) < 9:
        return 0

    # generate random data for 5 frames
    outputfile = sys._getframe().f_code.co_name + ".mp4"

    outputdata = np.random.random(size=(5, 480, 640, 3)) * 255
    outputdata = outputdata.astype(np.uint8)

    with skvideo.io.LibAVWriter(outputfile,verbosity=0) as writer:
        for i in range(5):
            writer.writeFrame(outputdata[i])
    os.remove(outputfile)

@unittest.skipIf(not skvideo._HAS_AVCONV, "LibAV required for this test.")
def test_LibAVWriter_Gray2RGBHack_Gray():
    _Gray2RGBHack_Helper('gray')


@unittest.skipIf(not skvideo._HAS_AVCONV, "LibAV required for this test.")
def test_LibAVWriter_Gray2RGBHack_ya8():
    _Gray2RGBHack_Helper('ya8')


@unittest.skipIf(not skvideo._HAS_AVCONV, "LibAV required for this test.")
def test_LibAVWriter_Gray2RGBHack_gray16le():
    _Gray2RGBHack_Helper('gray16le')


@unittest.skipIf(not skvideo._HAS_AVCONV, "LibAV required for this test.")
def test_LibAVWriter_Gray2RGBHack_gray16be():
    _Gray2RGBHack_Helper('gray16be')


def _Gray2RGBHack_Helper(pix_fmt):
    # skip if libav not installed or of the proper version
    if not skvideo._HAS_AVCONV:
        return 0
    if np.int(skvideo._LIBAV_MAJOR_VERSION) < 9:
        return 0

    # generate random data for 5 frames
    outputfile = sys._getframe().f_code.co_name + ".yuv"

    bits = 16 if pix_fmt[-4:][0:2] == '16' else 8
    outputdata = np.random.random(size=(1, 8, 8, 1)) * ((1<<bits)-1)

    if pix_fmt[0:2] == 'ya':
        outputdata = np.concatenate((outputdata, np.zeros_like(outputdata)), axis=3)
    if bits == 16:
        outputdata = outputdata.astype(np.uint16)
    else:
        outputdata = outputdata.astype(np.uint8)

    T,N,M,C = outputdata.shape

    with skvideo.io.LibAVWriter(outputfile,verbosity=0) as writer:
        for i in range(T):
            writer.writeFrame(outputdata[i])

    with skvideo.io.LibAVReader(outputfile, inputdict={'-s':'{}x{}'.format(M,N)}, verbosity=0) as reader:
        assert_equal(reader.getShape()[0:3],outputdata.shape[0:3])
        inputdata = np.empty(reader.getShape(), dtype=np.uint16)
        for i,frame in enumerate(reader.nextFrame()):
            inputdata[i] = frame * (1<<(bits-8))
    assert_array_less(np.abs(inputdata[:,:,:,0].astype('int32')- outputdata[:,:,:,0].astype('int32')),(1<<(bits-7)))
    os.remove(outputfile)

@unittest.skipIf(not skvideo._HAS_AVCONV, "LibAV required for this test.")
def test_LibAVReader_limit_vframe():
    # generate random data for 5 frames
    outputfile = sys._getframe().f_code.co_name + ".mp4"

    outputdata = np.random.random(size=(5, 16, 16, 3)) * 255
    outputdata = outputdata.astype(np.uint8)

    with skvideo.io.LibAVWriter(outputfile) as writer:
        for i in range(5):
            writer.writeFrame(outputdata[i])

    with skvideo.io.LibAVReader(outputfile, outputdict={'-vframes':'8'}, verbosity=0) as reader:
        T,M,N,C = reader.getShape()
    assert_equal(T,5)
    os.remove(outputfile)

@unittest.skipIf(not skvideo._HAS_AVCONV, "LibAV required for this test.")
def test_LibAVReader_raw_fps():
    # generate random data for 5 frames
    outputfile = sys._getframe().f_code.co_name + ".mp4"

    outputdata = np.random.random(size=(5, 16, 16, 3)) * 255
    outputdata = outputdata.astype(np.uint8)

    with skvideo.io.LibAVWriter(outputfile) as writer:
        for i in range(5):
            writer.writeFrame(outputdata[i])

    with skvideo.io.LibAVReader(outputfile,inputdict={'-r':'25'}, outputdict={'-r':'50'}, verbosity=0) as reader:
        T,M,N,C = reader.getShape()
    assert_equal(T,9)
    with skvideo.io.LibAVReader(outputfile,inputdict={'-r':'25'}, outputdict={'-r':'25'}, verbosity=0) as reader:
        T,M,N,C = reader.getShape()
    assert_equal(T,5)
    with skvideo.io.LibAVReader(outputfile,inputdict={'-r':'25'}, outputdict={'-r':'10'}, verbosity=0) as reader:
        T,M,N,C = reader.getShape()
    assert_equal(T,3)
    os.remove(outputfile)