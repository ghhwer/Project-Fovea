#! /usr/bin/env python
"""This program is based on YAD2K"""
"""Run a YOLO_v2 style detection model on test images."""
import argparse
import colorsys
import imghdr
import os
import random
import cv2

import numpy as np
from keras import backend as K
from keras.models import load_model
from PIL import Image, ImageDraw, ImageFont

from fovea.yolo_backend.yad2k.models.keras_yolo import yolo_eval, yolo_head

def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.size
    w, h = size
    scale = min(float(w)/iw, float(h)/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image

class yolo_handler():
    parser = argparse.ArgumentParser(
        description='Run a YOLO_v2 style detection model on test images..')
    parser.add_argument(
        'model_path',
        help='path to h5 model file containing body'
        'of a YOLO_v2 model')
    parser.add_argument(
        '-a',
        '--anchors_path',
        help='path to anchors file, defaults to yolo_anchors.txt',
        default='model_data/yolo_anchors.txt')
    parser.add_argument(
        '-c',
        '--classes_path',
        help='path to classes file, defaults to coco_classes.txt',
        default='model_data/coco_classes.txt')
    parser.add_argument(
        '-t',
        '--test_path',
        help='path to directory of test images, defaults to images/',
        default='images')
    parser.add_argument(
        '-o',
        '--output_path',
        help='path to output test images, defaults to images/out',
        default='images/out')
    parser.add_argument(
        '-s',
        '--score_threshold',
        type=float,
        help='threshold for bounding box scores, default .3',
        default=.3)
    parser.add_argument(
        '-iou',
        '--iou_threshold',
        type=float,
        help='threshold for non max suppression IOU, default .5',
        default=.5)

    def __init__(self,model_path,anchors_path,classes_path):
        model_path = os.path.expanduser(model_path)
        assert model_path.endswith('.h5'), 'Keras model must be a .h5 file.'
        anchors_path = os.path.expanduser(anchors_path)
        classes_path = os.path.expanduser(classes_path)

        self.sess = K.get_session()  # TODO: Remove dependence on Tensorflow session.

        with open(classes_path) as f:
            self.class_names = f.readlines()
        self.class_names = [c.strip() for c in self.class_names]

        with open(anchors_path) as f:
            self.anchors = f.readline()
            self.anchors = [float(x) for x in self.anchors.split(',')]
            self.anchors = np.array(self.anchors).reshape(-1, 2)

        self.yolo_model = load_model(model_path)

        # Verify model, anchors, and classes are compatible
        num_classes = len(self.class_names)
        num_anchors = len(self.anchors)
        # TODO: Assumes dim ordering is channel last
        model_output_channels = self.yolo_model.layers[-1].output_shape[-1]
        assert model_output_channels == num_anchors * (num_classes + 5), \
            'Mismatch between model and given anchor and class sizes. ' \
            'Specify matching anchors and classes with --anchors_path and ' \
            '--classes_path flags.'
        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Check if model is fully convolutional, assuming channel last order.
        self.model_image_size = self.yolo_model.layers[0].input_shape[1:3]
        self.is_fixed_size = self.model_image_size != (None, None)

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                    for x in range(len(self.class_names))]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                colors))
        random.seed(10101)  # Fixed seed for consistent colors across runs.
        random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
        random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        # TODO: Wrap these backend operations with Keras layers.
        yolo_outputs = yolo_head(self.yolo_model.output, self.anchors, len(self.class_names))
        self.input_image_shape = K.placeholder(shape=(2, ))
        self.boxes, self.scores, self.classes = yolo_eval(
            yolo_outputs,
            self.input_image_shape,
            score_threshold=0.5,
            iou_threshold=0.8)

    def do_predict(self, img, isPILBased = False):
        #Prepare image
        if not isPILBased:
            image = Image.fromarray(np.uint8(img))
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
        #Prediction
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]]
            })

        #Return Vector
        ret_vect = []
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]
            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5))
            left = max(0, np.floor(left + 0.5))
            bottom = min(image.size[1], np.floor(bottom + 0.5))
            right = min(image.size[0], np.floor(right + 0.5))
            ret_vect.append({'class':predicted_class,'rect':[top,bottom,left,right],'proba':score})

        return ret_vect

    def clean_up(self):
        self.sess.close()