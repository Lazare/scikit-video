import os
import time
import warnings

import numpy as np
from ..utils import *
import re
import sys


def getPixFmtInfos(pixfmt):
    if pixfmt not in bpplut:
        if pixfmt + endianess('le', 'be') in bpplut:
            pixfmt += endianess('le', 'be')
        else:
            raise ValueError(pixfmt + 'is not a valid pix_fmt')
    depth = np.int(bpplut[pixfmt][0])
    bpp = np.int(bpplut[pixfmt][1])
    bpc = bpp // depth
    if bpc == 8:
        dtype = np.dtype('u1')  # np.uint8
    elif bpc == 16:
        suffix = pixfmt[-2:]
        if suffix == 'le':
            dtype = np.dtype('<u2')
        elif suffix == 'be':
            dtype = np.dtype('>u2')
        else:
            raise ValueError(pixfmt + 'is strange it\'s 16 bits per channels but is neither "le" nor "be"')
    else:
        dtype = None
    return depth, bpp, dtype, pixfmt

def endianess(little='little', big='big'):
    return little if sys.byteorder == 'little' else big


def decodeFrameRate(framerate, default):
    if re.match(r'\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?', framerate) is None:
        raise ValueError("{} sould be numeric or numeric/numeric".format(framerate))
    parts = framerate.split('/')
    if len(parts) > 1:
        if np.float(parts[1]) == 0.:
            return default
        else:
            return np.float(parts[0]) / np.float(parts[1])
    else:
        return np.float(framerate)

def decodeFrameSize(size):
    parts = size.split('x')
    return np.int(parts[0]), np.int(parts[1])

def dict2Args(dict):
    args = []
    for key in dict.keys():
        args.append(key)
        args.append(dict[key])
    return args

def getRotation(vidDict):
    if ('tag' in vidDict):
        tagdata = vidDict['tag']
        if not isinstance(tagdata, list):
            tagdata = [tagdata]

        for tags in tagdata:
            if tags['@key'] == 'rotate':
                return tags['@value']
    return '0'


class VideoAbstract(object):
    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

class VideoReaderAbstract(VideoAbstract):
    """Reads frames
    """

    INFO_AVERAGE_FRAMERATE = None #"avg_frame_rate"
    INFO_WIDTH = None #"width"
    INFO_HEIGHT = None #"height"
    INFO_PIX_FMT = None #"pix_fmt"
    INFO_DURATION = None #"duration"
    INFO_NB_FRAMES = None #"nb_frames"
    DEFAULT_FRAMERATE = 25.
    DEFAULT_INPUT_PIX_FMT = "yuvj444p"
    OUTPUT_METHOD = None # "rawvideo"

    def __init__(self, filename=None, inputdict=None, outputdict=None, verbosity=0):
        """Initializes FFmpeg in reading mode with the given parameters

        During initialization, additional parameters about the video file
        are parsed using :func:`skvideo.io.ffprobe`. Then FFmpeg is launched
        as a subprocess. Parameters passed into inputdict are parsed and
        used to set as internal variables about the video. If the parameter,
        such as "Height" is not found in the inputdict, it is found through
        scanning the file's header information. If not in the header, ffprobe
        is used to decode the file to determine the information. In the case
        that the information is not supplied and connot be inferred from the
        input file, a ValueError exception is thrown.

        Parameters
        ----------
        filename : string
            Video file path

        inputdict : dict
            Input dictionary parameters, i.e. how to interpret the input file.

        outputdict : dict
            Output dictionary parameters, i.e. how to encode the data
            when sending back to the python process.

        Returns
        -------
        none

        """

        if not inputdict:
            inputdict = {}

        if not outputdict:
            outputdict = {}

        # define the kind of input
        protocols = self._getSupportedInputProtocols()
        if protocols is NotImplemented or len(protocols) < 1:
            isFile =True
        elif filename[0:5] == 'file:':
            filename = filename[5:]
            isFile = True
        elif re.match('^{}:'.format(b'|'.join(protocols)), filename) is not None:
            isFile = False
        else:
            isFile = True

        self._filename = filename
        _, extension = os.path.splitext(filename)

        if isFile:
            isRawWebCam = re.match(r'/dev/video\d+', filename) is not None
            size = 0 if isRawWebCam else os.path.getsize(filename)
            israw = isRawWebCam or str.encode(extension) in [b".raw", b".yuv"]
            if not israw:
                decoders = self._getSupportedDecoders()
                if decoders != NotImplemented:
                    # check that the extension makes sense
                    assert str.encode(
                        extension).lower() in decoders, "Unknown decoder extension: " + extension.lower()

        else:
            israw = False

        if israw:
            viddict = {}
        else:
            probeInfo = self._probe()
            if "video" in probeInfo:
                viddict = probeInfo["video"]
            else:
                viddict = {}


        #---------- input options --------------------

        if ("-r" in inputdict):
            inputfps = decodeFrameRate(inputdict["-r"], default=self.DEFAULT_FRAMERATE)
        elif self.INFO_AVERAGE_FRAMERATE in viddict:
            inputfps = decodeFrameRate(viddict[self.INFO_AVERAGE_FRAMERATE], default=self.DEFAULT_FRAMERATE)
        else:
            inputfps = self.DEFAULT_FRAMERATE
            if israw and "-r" in outputdict: # if resampling of the framerate is activated in the output set the input framerate to default for raw files
                inputdict["-r"] = inputfps

        # if we don't have width or height at all, raise exception
        if ("-s" in inputdict):
            self.inputwidth, self.inputheight = decodeFrameSize(inputdict["-s"])
        elif ((self.INFO_WIDTH in viddict) and (self.INFO_HEIGHT in viddict)):
            self.inputwidth = np.int(viddict[self.INFO_WIDTH])
            self.inputheight = np.int(viddict[self.INFO_HEIGHT])
        else:
            raise ValueError(
                "No way to determine width or height from video. Need `-s` in `inputdict`. Consult documentation on I/O.")

        # smartphone video data is weird
        # smartphone recordings seem to store data about rotations
        # in tag format. Just swap the width and height
        if getRotation(viddict) in ['90','270']:
            self.inputwidth, self.inputheight = self.inputheight, self.inputwidth

        if ("-pix_fmt" in inputdict):
            input_pix_fmt = inputdict["-pix_fmt"]
        elif (self.INFO_PIX_FMT in viddict):
            input_pix_fmt = viddict[self.INFO_PIX_FMT]
        else:
            input_pix_fmt = self.DEFAULT_INPUT_PIX_FMT
            if verbosity != 0:
                warnings.warn("No input color space detected. Assuming {}.".format(self.DEFAULT_INPUT_PIX_FMT), UserWarning)

        if israw:
            inputdict['-pix_fmt'] = input_pix_fmt
        self.inputdepth, self.bpp, _, _ = getPixFmtInfos(input_pix_fmt)

        if self.INFO_NB_FRAMES in viddict:
            inputFrameNum = np.int(viddict[self.INFO_NB_FRAMES])
        elif israw and size is not None:
            inputFrameNum = np.int(size / (self.inputwidth * self.inputheight * (self.bpp / 8.0)))
        else:
            inputFrameNum = None

        #--------- output options --------------------

        if "-r" in outputdict:
            outputfps = decodeFrameRate(outputdict["-r"], default=self.DEFAULT_FRAMERATE)
            if self.INFO_DURATION in viddict:
                duration = np.float(viddict[self.INFO_DURATION])
            else:
                duration = float(inputFrameNum) / float(inputfps)
            self.outputFrameNum = self._getResampledNumberOfFrames(inputfps, outputfps, duration)
        else:
            self.outputFrameNum = inputFrameNum

        if "-vframes" in outputdict:
            self.outputFrameNum = np.int(outputdict["-vframes"]) if self.outputFrameNum is None else min(self.outputFrameNum, np.int(outputdict["-vframes"]))
        elif "-frames" in outputdict:
            self.outputFrameNum = np.int(outputdict["-frames"]) if self.outputFrameNum is None else min(self.outputFrameNum, np.int(outputdict["-frames"]))
        elif self.outputFrameNum is None:
            self.outputFrameNum = self._probCountFrames()
            if verbosity != 0:
                warnings.warn(
                    "Cannot determine frame count. Scanning input file, this is slow when repeated many times. Need `-vframes` in inputdict. Consult documentation on I/O.",
                    UserWarning)

        if '-f' not in outputdict:
            outputdict['-f'] = self.OUTPUT_METHOD

        if '-pix_fmt' not in outputdict:
            outputdict['-pix_fmt'] = "rgb24"

        if '-s' in outputdict:
            self.outputwidth, self.outputheight = decodeFrameSize(outputdict["-s"])
        else:
            self.outputwidth = self.inputwidth
            self.outputheight = self.inputheight

        self.outputdepth ,self.outputbpp, self.dtype, outputdict['-pix_fmt'] = getPixFmtInfos(outputdict['-pix_fmt'])
        if self.dtype is None:
            raise ValueError(outputdict['-pix_fmt'] + 'is not a valid pix_fmt for numpy conversion')

        self._createProcess(inputdict, outputdict, verbosity)

    def _createProcess(self, inputdict, outputdict, verbosity):
        pass

    def _getResampledNumberOfFrames(self, inputfps, outputfps, duration):
        return np.int(round(outputfps * (duration - 1.0 / inputfps)) + 2)

    def _probCountFrames(self):
        return NotImplemented

    def _probe(self):
        pass

    def _getSupportedDecoders(self):
        return NotImplemented

    def _getSupportedInputProtocols(self):
        return NotImplemented

    def getShape(self):
        """Returns a tuple (T, M, N, C)

        Returns the video shape in number of frames, height, width, and channels per pixel.
        """

        return self.outputFrameNum, self.outputheight, self.outputwidth, self.outputdepth

    def close(self):
        if self._proc is not None and self._proc.poll() is None:
            self._proc.stdin.close()
            self._proc.stdout.close()
            self._proc.stderr.close()
            self._terminate(0.2)
        self._proc = None

    def _terminate(self, timeout=1.0):
        """ Terminate the sub process.
        """
        # Check
        if self._proc is None:  # pragma: no cover
            return  # no process
        if self._proc.poll() is not None:
            return  # process already dead
        # Terminate process
        self._proc.terminate()
        # Wait for it to close (but do not get stuck)
        etime = time.time() + timeout
        while time.time() < etime:
            time.sleep(0.01)
            if self._proc.poll() is not None:
                break

    def _readFrame(self):
        # Init and check
        framesize = self.outputdepth * self.outputwidth * self.outputheight
        assert self._proc is not None

        try:
            # Read framesize bytes
            arr = np.frombuffer(self._proc.stdout.read(framesize * self.dtype.itemsize), dtype=self.dtype)
            if len(arr) == 0:
                return None
            assert len(arr) == framesize
        except Exception as err:
            self._terminate()
            err1 = str(err)
            raise RuntimeError("%s" % (err1,))
        self._lastread = arr.reshape((self.outputheight, self.outputwidth, self.outputdepth))
        return self._lastread

    def nextFrame(self):
        """Yields frames using a generator

        Returns T ndarrays of size (M, N, C), where T is number of frames,
        M is height, N is width, and C is number of channels per pixel.

        """
        if self.outputFrameNum is None:
            while True:
                frame = self._readFrame()
                if frame is None:
                    break
                else:
                    yield frame
        else:
            for i in range(self.outputFrameNum):
                frame = self._readFrame()
                if frame is None:
                    break
                else:
                    yield frame


class VideoWriterAbstract(VideoAbstract):
    """Writes frames

    this class provides sane initializations for the default case.
    """
    NEED_RGB2GRAY_HACK = False
    DEFAULT_OUTPUT_PIX_FMT = "yuvj444p"

    def __init__(self, filename, inputdict=None, outputdict=None, verbosity=0):
        """Prepares parameters

        Does not instantiate the an FFmpeg subprocess, but simply
        prepares the required parameters.

        Parameters
        ----------
        filename : string
            Video file path for writing

        inputdict : dict
            Input dictionary parameters, i.e. how to interpret the data coming from python.

        outputdict : dict
            Output dictionary parameters, i.e. how to encode the data
            when writing to file.

        Returns
        -------
        none

        """
        self.DEVNULL = open(os.devnull, 'wb')

        filename = os.path.abspath(filename)
        _, extension = os.path.splitext(filename)

        self._filename = filename

        # check that the extension makes sense
        encoders = self._getSupportedEncoders()
        if encoders != NotImplemented:
            assert str.encode(
                extension).lower() in encoders, "Unknown encoder extension: " + extension.lower()

        basepath, _ = os.path.split(filename)

        # check to see if filename is a valid file location
        assert os.access(basepath, os.W_OK), "Cannot write to directory: " + basepath

        if not inputdict:
            inputdict = {}

        if not outputdict:
            outputdict = {}

        self.inputdict = inputdict
        self.outputdict = outputdict
        self.verbosity = verbosity

        if "-f" not in self.inputdict:
            self.inputdict["-f"] = "rawvideo"
        self.warmStarted = False
        self._instancePrepareData = self._prepareData

    def _warmStart(self, M, N, C, dtype):
        self.warmStarted = True

        if "-pix_fmt" not in self.inputdict:
            # check the number channels to guess
            if dtype.kind == 'u' and dtype.itemsize == 2:
                suffix = 'le' if dtype.byteorder == '<' else 'be'
                if C == 1:
                    if self.NEED_RGB2GRAY_HACK:
                        self.inputdict["-pix_fmt"] = "rgb48" + suffix
                        self.rgb2grayhack = True
                        C = 3
                    else:
                        self.inputdict["-pix_fmt"] = "gray16" + suffix
                elif C == 2:
                    self.inputdict["-pix_fmt"] = "ya16" + suffix
                elif C == 3:
                    self.inputdict["-pix_fmt"] = "rgb48" + suffix
                elif C == 4:
                    self.inputdict["-pix_fmt"] = "rgba64" + suffix
                else:
                    raise NotImplemented
            else:
                if C == 1:
                    if self.NEED_RGB2GRAY_HACK:
                        self.inputdict["-pix_fmt"] = "rgb24"
                        self.rgb2grayhack = True
                        C = 3
                    else:
                        self.inputdict["-pix_fmt"] = "gray"
                elif C == 2:
                    self.inputdict["-pix_fmt"] = "ya8"
                elif C == 3:
                    self.inputdict["-pix_fmt"] = "rgb24"
                elif C == 4:
                    self.inputdict["-pix_fmt"] = "rgba"
                else:
                    raise NotImplemented

        self.bpp = bpplut[self.inputdict["-pix_fmt"]][1]
        self.inputNumChannels = bpplut[self.inputdict["-pix_fmt"]][0]
        bitpercomponent = self.bpp // self.inputNumChannels
        if bitpercomponent == 8:
            self.dtype = np.dtype('u1')  # np.uint8
        elif bitpercomponent == 16:
            suffix = self.inputdict['-pix_fmt'][-2:]
            if suffix == 'le':
                self.dtype = np.dtype('<u2')
            elif suffix == 'be':
                self.dtype = np.dtype('>u2')
        else:
            raise ValueError(self.inputdict['-pix_fmt'] + 'is not a valid pix_fmt for numpy conversion')

        assert self.inputNumChannels == C, "Failed to pass the correct number of channels %d for the pixel format %s." % (
        self.inputNumChannels, self.inputdict["-pix_fmt"])

        if ("-s" in self.inputdict):
            widthheight = self.inputdict["-s"].split('x')
            self.inputwidth = np.int(widthheight[0])
            self.inputheight = np.int(widthheight[1])
        else:
            self.inputdict["-s"] = str(N) + "x" + str(M)
            self.inputwidth = N
            self.inputheight = M

        # prepare output parameters, if raw
        _, extension = os.path.splitext(self._filename)
        if extension == ".yuv":
            if "-pix_fmt" not in self.outputdict:
                self.outputdict["-pix_fmt"] = self.DEFAULT_OUTPUT_PIX_FMT
                if self.verbosity > 0:
                    warnings.warn("No output color space provided. Assuming {}.".format(self.DEFAULT_OUTPUT_PIX_FMT), UserWarning)

        self._createProcess(self.inputdict, self.outputdict, self.verbosity)

    def _createProcess(self, inputdict, outputdict, verbosity):
        self._cmd = ''
        pass

    def _prepareData(self, data):
        return data # general case : do nothing

    def close(self):
        """Closes the video and terminates FFmpeg process

        """
        if self._proc is None:  # pragma: no cover
            return  # no process
        if self._proc.poll() is not None:
            return  # process already dead
        if self._proc.stdin:
            self._proc.stdin.close()
        self._proc.wait()
        self._proc = None
        self.DEVNULL.close()

    def writeFrame(self, im):
        """Sends ndarray frames to FFmpeg

        """
        vid = vshape(im)
        T, M, N, C = vid.shape
        if not self.warmStarted:
            self._warmStart(M, N, C, im.dtype)

        vid = vid.clip(0, (1 << (self.dtype.itemsize << 3)) - 1).astype(self.dtype)

        vid = self._instancePrepareData(vid)
        T, M, N, C = vid.shape # in case of hack in instancePrepareData to change the image shape (gray2RGB in libAV for exemple)

        # Check size of image
        if M != self.inputheight or N != self.inputwidth:
            raise ValueError('All images in a movie should have same size')
        if C != self.inputNumChannels:
            raise ValueError('All images in a movie should have same '
                             'number of channels')

        assert self._proc is not None  # Check status

        # Write
        try:
            self._proc.stdin.write(vid.tostring())
        except IOError as e:
            # Show the command and stderr from pipe
            msg = '{0:}\n\nFFMPEG COMMAND:\n{1:}\n\nFFMPEG STDERR ' \
                  'OUTPUT:\n'.format(e, self._cmd)
            raise IOError(msg)

    def _getSupportedEncoders(self):
        return NotImplemented
