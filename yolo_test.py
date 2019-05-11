import cv2
from fovea.artist.opencv_artist import drawPrediction_yolo
from fovea.artist.opencv_artist import threaded_source_manipulation
from fovea.yolo_backend.yolo_run import yolo_handler

def get_image_from_camera(src,vars):
    image = vars['cam'].read()[1]
    if image is not None:
        src = image
        return src

def draw_img(src,vars):
    if not (vars['yolo_data'] is None or vars['yolo_data'] == []):
        drawPrediction_yolo(src,out)
    if src is not None:
        cv2.imshow('image',src)
        if cv2.waitKey(1) == 27:
            vars['kill_program'] = True
    return -1   

#Load Pre-trained YOLO Model
y = yolo_handler('model_data/keras/itau_keras.h5','model_data/keras/itau_keras_anchors.txt','model_data/keras/itau_classes.txt')

staring_vars = [{'name':'cam','var':cv2.VideoCapture(0)},{'name':'yolo_data', 'var':None},{'name':'kill_program','var':False}]  #Threaded ambient vars
functions = [get_image_from_camera,draw_img] #functions to run on a loop

#A helper tool to get multi-thread sync
tsm = threaded_source_manipulation(functions,staring_vars)

#Main program loop
while True:
    image = tsm.get_src()
    if image is not None:
        out = y.do_predict(image)
        tsm.push_variable_to_ambient('yolo_data',out)
    if tsm.vars['kill_program'] == True:
        break

# Clean up
cv2.destroyAllWindows()
tsm.clean_up()
tsm.vars['cam'].release() #Closes video file or capturing device.
y.clean_up()