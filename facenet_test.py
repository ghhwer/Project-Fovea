import cv2
from fovea.artist.opencv_artist import drawDetection_facenet
from fovea.artist.opencv_artist import threaded_source_manipulation
from fovea.facenet_backend.facenet_embedding import facenet_handler

def get_image_from_camera(src,vars):
    image = vars['cam'].read()[1]
    if image is not None:
        src = image
        return src

def draw_img(src,vars):
    if not (vars['facenet_data'] is None or vars['facenet_data'] == []):
        drawDetection_facenet(src,out)
    if src is not None:
        cv2.imshow('image',src)
        if cv2.waitKey(1) == 27:
            vars['kill_program'] = True
    return -1   

#Load Pre-trained YOLO Model
f = facenet_handler('model_data')

staring_vars = [{'name':'cam','var':cv2.VideoCapture(0)},{'name':'facenet_data', 'var':None},{'name':'kill_program','var':False}]  #Threaded ambient vars
functions = [get_image_from_camera,draw_img] #functions to run on a loop

#A helper tool to get multi-thread sync
tsm = threaded_source_manipulation(functions,staring_vars)

#Main program loop
while True:
    image = tsm.get_src()
    if image is not None:
        out = f.do_predict(image)
        tsm.push_variable_to_ambient('facenet_data',out)
    if tsm.vars['kill_program'] == True:
        break

# Clean up
cv2.destroyAllWindows()
tsm.clean_up()
tsm.vars['cam'].release() #Closes video file or capturing device.
y.clean_up()