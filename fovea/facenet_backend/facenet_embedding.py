import os
import json
import cv2
import copy

import tensorflow as tf
import keras.backend.tensorflow_backend as K

from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.layers import Dense, Input, Dropout, MaxPooling2D, Conv2D, Flatten
from keras.utils import np_utils

from keras.models import model_from_json
from keras.models import load_model
from keras import optimizers


import dlib

import numpy as np
from fovea.facenet_backend.model_code.inception_resnet_v1 import *

class facenet_handler():
    def __init__(self,rPath,
                predictor_5_face_path = '/dlib/shape_predictor_5_face_landmarks.dat',
                predictor_68_face_path = '/dlib/shape_predictor_68_face_landmarks.dat',
                facenet_path='/keras/facenet_keras.h5',
                image_size=160, debug=False):
        self.facenet_model_path = rPath+facenet_path
        self.predictor_5_face_path = rPath+predictor_5_face_path
        self.predictor_68_face_path = rPath+predictor_68_face_path
        self._load_detector()
        self._load_facenet()
        self.debug = debug
        self.IM_SIZE = image_size

    #RESOURCE LOADING AND SAVING
    def _load_detector(self):
        self.detector = dlib.get_frontal_face_detector()
        self.sp_5_face = dlib.shape_predictor(self.predictor_5_face_path)
        self.sp_68_face = dlib.shape_predictor(self.predictor_68_face_path)

    def _load_facenet(self):
        self.facenet_model = InceptionResNetV1()
        self.facenet_model.load_weights(self.facenet_model_path)
        self.graph = tf.get_default_graph()

    #Image detection and classification routines

    #Dimention Adjust
    def _facenet_prewhiten(self,x):
        if x.ndim == 4:
            axis = (1, 2, 3)
            size = x[0].size
        elif x.ndim == 3:
            axis = (0, 1, 2)
            size = x.size
        else:
            raise ValueError('Dimension should be 3 or 4')

        mean = np.mean(x, axis=axis, keepdims=True)
        std = np.std(x, axis=axis, keepdims=True)
        std_adj = np.maximum(std, 1.0/np.sqrt(size))
        y = (x - mean) / std_adj
        return y

    #FaceNet embeddings
    def _facenet_emb(self, image):
        with self.graph.as_default():
            x = self._facenet_prewhiten(image)
            y = self.facenet_model.predict(x)
            return y[0]

    #Detector
    def _do_align(self,img):
        dets = self.detector(img, 1)
        num_faces = len(dets)
        if num_faces == 0:
            return None,None
        else:
            rects = dets
        # Find the 5 face landmarks we need to do the alignment.
        faces = dlib.full_object_detections()
        for detection in dets:
            faces.append(self.sp_5_face(img, detection))

        # Get the aligned face images
        # Optionally:
        # images = dlib.get_face_chips(img, faces, size=160, padding=0.25)
        images = dlib.get_face_chips(img, faces, size=160, padding=0.5)
        return images, rects

    #Classifier
    def _network_pass(self, img):
        img_c = copy.deepcopy(img)
        emb = self._facenet_emb(img_c)
        return emb

    def do_landmark_find(self,img):
        # Detect faces in the image
        faceRects = self.detector(img, 0)
        num_faces = len(faceRects)
        if num_faces == 0:
            return

        # List to store landmarks of all detected faces
        landmarksAll = []
        newRect = []

        # Loop over all detected face rectangles
        for i in range(0, len(faceRects)):
            newRect = dlib.rectangle(int(faceRects[i].left()),int(faceRects[i].top()),
            int(faceRects[i].right()),int(faceRects[i].bottom()))

        if newRect is None:
            return

        # For every face rectangle, run landmarkDetector
        landmarks = self.sp_68_face(img, newRect)

        # Store landmarks for current face
        landmarksAll.append(landmarks)
        return landmarks

    def do_predict(self, im):
        imgs, rects = self._do_align(im)
        face_strcts = []
        if imgs is not None and rects is not None:
            if len(imgs) == len(rects):
                i = 0
                for x in imgs:
                    face = {'rect':None,'id':-1,'emb':None}
                    img = x
                    rect = rects[i]
                    face['rect'] = [rect.top(),rect.bottom(),rect.left(),rect.right()]
                    frameAlign = cv2.resize(img,(self.IM_SIZE,self.IM_SIZE))
                    i+=1
                    if frameAlign is not None:
                        imgs = np.array([frameAlign])
                        r = self._network_pass(imgs)
                        face['emb'] = r
                        face_strcts.append(face)
        else:
            return None
        return face_strcts
