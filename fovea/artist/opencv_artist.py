import cv2
import copy

def drawPrediction_yolo(im, struct, text_bg_color=(0,0,255), outline_color=(0,0,255)):
    if struct == None:
        return
    struct = list(reversed(struct))
    for f in struct:
        rect = f['rect']
        label = f['class']
        proba = f['proba']
        x1 = int(rect[2])
        y1 = int(rect[0])
        x2 = int(rect[3])
        y2 = int(rect[1])
        #Rectangle 1 Name and its proba
        im = cv2.rectangle(im,(x1,y1),(x2,y2),outline_color,2)
        txt_bx = cv2.getTextSize(str(label)+' '+str('%0.2f'%(float(proba)*100)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        im = cv2.rectangle(im,(x1+txt_bx[0][0]+25,y1-txt_bx[0][1]-10),(x1,y1),tuple(text_bg_color),cv2.FILLED)
        im = cv2.putText(im, str(label)+' '+str('%0.2f'%(float(proba)*100))+'%',(x1+5,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

def drawDetection_facenet(im, struct, text_bg_color=(0,0,255), outline_color=(0,0,255)):
    if struct == None:
        return
    for f in struct:
        rect = f['rect']
        x1 = int(rect[2])
        y1 = int(rect[0])
        x2 = int(rect[3])
        y2 = int(rect[1])
        #Rectangle 1 Name and its proba
        im = cv2.rectangle(im,(x1,y1),(x2,y2),outline_color,2)


class threaded_source_manipulation():
    def __init__(self,source_functions,staring_vars):
        "Functions will run first to last"
        from threading import Thread
        i=0
        self.flags = []
        self.src = None
        self.vars = {}
        for x in staring_vars:
            self.push_variable_to_ambient(x['name'],x['var'])
        for x in source_functions:
            self.flags.append({'stop':False, 'wait':False})
            Thread(target=self._serve, args=(i,x)).start()
            i+=1

    def get_src(self):
        return copy.deepcopy(self.src)

    def push_variable_to_ambient(self,var_name, var):
        self.vars[var_name] = var
    
    def _serve(self,id,function):
        while self.flags[id]['stop'] is False:
            if self.flags[id]['wait'] is False:
                o = function(self.src,self.vars)
                if o is not -1:
                    self.src = o

    def clean_up(self):
        for x in self.flags:
            x['stop'] = True

