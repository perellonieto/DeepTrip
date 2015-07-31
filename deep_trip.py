#!/usr/bin/python
import argparse
import os
from cStringIO import StringIO
import numpy as np
import scipy.ndimage as nd
import PIL.Image
from google.protobuf import text_format

import caffe

model_path = './models/bvlc_googlenet/' # substitute your path here
net_fn   = model_path + 'deploy.prototxt'
param_fn = model_path + 'bvlc_googlenet.caffemodel'

def load_net(model_path, net_fn, param_fn):
    # Patching model to be able to compute gradients.
    # Note that you can also manually add "force_backward: true" line to
    # "deploy.prototxt".
    model = caffe.io.caffe_pb2.NetParameter()
    text_format.Merge(open(net_fn).read(), model)
    model.force_backward = True
    open('tmp.prototxt', 'w').write(str(model))

    net = caffe.Classifier('tmp.prototxt', param_fn,
                           mean = np.float32([104.0, 116.0, 122.0]), # ImageNet mean, training set dependent
                           channel_swap = (2,1,0)) # the reference model has channels in BGR order instead of RGB
    return net

# a couple of utility functions for converting to and from Caffe's input image
# layout
def preprocess(net, img):
    return np.float32(np.rollaxis(img, 2)[::-1]) - net.transformer.mean['data']
def deprocess(net, img):
    return np.dstack((img + net.transformer.mean['data'])[::-1])

def make_step(net, step_size=1.5, end='inception_4c/output', jitter=32,
              clip=True):
    '''Basic gradient ascent step.'''

    src = net.blobs['data'] # input image is stored in Net's 'data' blob
    dst = net.blobs[end]

    ox, oy = np.random.randint(-jitter, jitter+1, 2)
    src.data[0] = np.roll(np.roll(src.data[0], ox, -1), oy, -2) # jitter shift

    net.forward(end=end)
    dst.diff[:] = dst.data  # specify the optimization objective
    net.backward(start=end)
    g = src.diff[0]
    # apply normalized ascent step to the input image
    src.data[:] += step_size/np.abs(g).mean() * g

    src.data[0] = np.roll(np.roll(src.data[0], -ox, -1), -oy, -2) # unshift img

    if clip:
        bias = net.transformer.mean['data']
        src.data[:] = np.clip(src.data, -bias, 255-bias)

def deepdream(net, base_img, iter_n=10, octave_n=4, octave_scale=1.4,
        end='inception_4c/output', clip=True, **step_params):
    # prepare base images for all octaves
    octaves = [preprocess(net, base_img)]
    for i in xrange(octave_n-1):
        octaves.append(nd.zoom(octaves[-1],
                       (1, 1.0/octave_scale,1.0/octave_scale),
                        order=1))

    src = net.blobs['data']
    detail = np.zeros_like(octaves[-1]) # allocate image for produced details
    for octave, octave_base in enumerate(octaves[::-1]):
        h, w = octave_base.shape[-2:]
        if octave > 0:
            # upscale details from the previous octave
            h1, w1 = detail.shape[-2:]
            detail = nd.zoom(detail, (1, 1.0*h/h1,1.0*w/w1), order=1)

        src.reshape(1,3,h,w) # resize the network's input image size
        src.data[0] = octave_base+detail
        for i in xrange(iter_n):
            make_step(net, end=end, clip=clip, **step_params)

            # visualization
            vis = deprocess(net, src.data[0])
            if not clip: # adjust image contrast if clipping is disabled
                vis = vis*(255.0/np.percentile(vis, 99.98))
            print octave, i, end, vis.shape

        # extract details produced on the current octave
        detail = src.data[0]-octave_base
    # returning the resulting image
    return deprocess(net, src.data[0])

def main(net, output_folder, img_name, octaves, octavescale, itern,
                iterplayer, jitter, stepsize):
    print img_name
    if img_name == None:
        frame = np.random.randint(100,200,size=(720,1280,3))
    else:
        frame = np.float32(PIL.Image.open(img_name))

    PIL.Image.fromarray(np.uint8(frame)).save(
                os.path.join(output_folder, "{:0>9d}.jpg".format(0)))

    h, w = frame.shape[:2]
    s = 0.05 # scale coefficient
    if octaves is None: octaves = 1
    if octavescale is None: octavescale = 1.5
    if itern is None: itern = 10
    if iterplayer is None: iterplayer = 1
    if jitter is None: jitter = 32
    if stepsize is None: stepsize = 1.5

    for i, layer in enumerate(net.blobs.keys(), start=0):
        for j in range(iterplayer):
            try:
                frame = deepdream(net, frame, iter_n=itern,
                                  step_size=stepsize, octave_n=octaves,
                                  octave_scale=octavescale, jitter=jitter,
                                  end=layer)
            except ValueError:
                print("ValueError: layer {} is not in the list".format(layer))
                break
            else:
                PIL.Image.fromarray(np.uint8(frame)).save(
                        os.path.join(output_folder,
                        "{:0>3d}{:0>5d}.jpg".format(i,j)))
                frame = nd.affine_transform(frame, [1-s,1-s,1],
                                            [h*s/2,w*s/2,0], order=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DeepTrip')
    parser.add_argument('-i','--input', help='Input image',
                        type=str, required=False)
    parser.add_argument('-o','--output',help='Output directory', required=True)
    parser.add_argument('-oct','--octaves',help='Octaves. Default: 4',
                        type=int, required=False)
    parser.add_argument('-octs','--octavescale',
                        help='Octave Scale. Default: 1.4', type=float,
                        required=False)
    parser.add_argument('-itr','--iterations',help='Iterations. Default: 10',
                        type=int, required=False)
    parser.add_argument('-itrpl','--iterplayer',
                        help='Iterations per layer. Default: 10', type=int,
                        required=False)
    parser.add_argument('-j','--jitter',help='Jitter. Default: 32', type=int,
                        required=False)
    parser.add_argument('-s','--stepsize',help='Step Size. Default: 1.5',
                        type=float, required=False)

    args = parser.parse_args()

    net = load_net(model_path, net_fn, param_fn)
    main(net, args.output, args.input, args.octaves,
         args.octavescale, args.iterations, args.iterplayer, args.jitter,
         args.stepsize)
