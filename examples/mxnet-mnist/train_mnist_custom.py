"""
Train mnist, see more explanation at http://mxnet.io/tutorials/python/mnist.html
"""
import os
import argparse
import logging
logging.basicConfig(level=logging.DEBUG)
from common import find_mxnet, fit
from common.util import download_file
import mxnet as mx
import numpy as np
import gzip, struct


def make_mlp_network(num_classes=10, **kwargs):
    """
    Make the network for the task. This is a simple multi-layer perceptron
    """
    hidden_layers = 3
    num_hidden = 128
    act_type='relu'
    if kwargs is not None:
        hidden_layers = kwargs.get('hidden_layers', 3)
        num_hidden = kwargs.get('num_hidden', 128)
        act_type = kwargs.get('act_type', 'relu')
    # Input Layer
    data = mx.symbol.Variable('data')
    out = mx.sym.Flatten(data=data)
    # Hidden Layers
    for layer in range(hidden_layers):
        fc = mx.symbol.FullyConnected(data=out, name='fc{}'.format(layer), num_hidden=num_hidden)
        out = mx.symbol.Activation(data=fc, name='{}{}'.format(act_type, layer), act_type=act_type)
    # Softmax output
    fc = mx.symbol.FullyConnected(data=out, name='fc{}'.format(hidden_layers), num_hidden=num_classes)
    out = mx.symbol.SoftmaxOutput(data = fc, name='softmax')
    return out

def read_data(label, image):
    """
    download and read data into numpy
    """
    base_url = 'http://yann.lecun.com/exdb/mnist/'
    with gzip.open(download_file(base_url+label, os.path.join('data',label))) as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        label = np.fromstring(flbl.read(), dtype=np.int8)
    with gzip.open(download_file(base_url+image, os.path.join('data',image)), 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        image = np.fromstring(fimg.read(), dtype=np.uint8).reshape(len(label), rows, cols)
    return (label, image)


def to4d(img):
    """
    reshape to 4D arrays
    """
    return img.reshape(img.shape[0], 1, 28, 28).astype(np.float32)/255

def get_mnist_iter(args, kv):
    """
    create data iterator with NDArrayIter
    """
    (train_lbl, train_img) = read_data(
            'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz')
    (val_lbl, val_img) = read_data(
            't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz')
    train = mx.io.NDArrayIter(
        to4d(train_img), train_lbl, args.batch_size, shuffle=True)
    val = mx.io.NDArrayIter(
        to4d(val_img), val_lbl, args.batch_size)
    return (train, val)

if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser(description="train mnist",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--num-classes', type=int, default=10,
                        help='the number of classes')
    parser.add_argument('--num-examples', type=int, default=60000,
                        help='the number of training examples')
    fit.add_fit_args(parser)
    parser.set_defaults(
        # network
        network        = 'mlp',
        num_layers     = 3,
        num_hidden     = 128,
        act_type       = 'relu',
        # train
        gpus           = None,
        batch_size     = 64,
        disp_batches   = 100,
        num_epochs     = 20,
        lr             = .05,
        lr_step_epochs = '10',
        kill_time      = 5,
    )
    args = parser.parse_args()

    # load network
    sym = make_mlp_network(**vars(args))

    # train
    fit.fit(args, sym, get_mnist_iter)
