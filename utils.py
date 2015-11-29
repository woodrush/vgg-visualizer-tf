import numpy as np
import scipy.misc
from datetime import datetime as dt
import argparse
from models import VGG16, I2V

mean = np.array([103.939, 116.779, 123.68], dtype=np.float32)

def sub_mean(img):
    for i in range(3):
        img[0,:,:,i] -= mean[i]
    return img

def read_image(path, w=0):
    img = scipy.misc.imread(path)
    # Resize if ratio is specified
    if w > 0:
        r = w / np.float32(img.shape[1])
        img = scipy.misc.imresize(img, (int(img.shape[0]*r), int(img.shape[1]*r)))
    img = img.astype(np.float32)
    img = img[None, ...]
    return img

def parseArgs():
    parser = argparse.ArgumentParser(
        description='VGG Visualizer in TensorFlow')
    parser.add_argument('--content', '-c', default='images/sd.jpg',
                        help='Content image path')
    parser.add_argument('--model', '-m', default='vgg',
                        help='Model type (vgg, i2v)')
    parser.add_argument('--modelpath', '-mp', default='vgg',
                        help='Model file path')
    parser.add_argument('--width', '-w', default=-1, type=int,
                        help='Input image resize target width (default: no resizing. Smaller images run faster and with fewer memory)')
    parser.add_argument('--maxfilters', '-x', default=20, type=int,
                        help='Maximum Nnumber of filters to show per layer')
    args = parser.parse_args()
    return args.content, args.modelpath, args.model, args.maxfilters

def getModel(image, params_path, model):
    if model == 'vgg':
        return VGG16(image, params_path)
    elif model == 'i2v':
        return I2V(image, params_path)
    else:
        print 'Invalid model name: use `vgg` or `i2v`'
        return None