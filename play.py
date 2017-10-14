import pdb
import skvideo.io
import skvideo.datasets
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
#from pyqtgraph.ptime import time as ptime
import time
import numpy as np
import cv2
import pdb
import os


def raise_alarm():
    """
    Flash box or text
    """
    print('look back')


class Gaze(object):

    def __init__(self, hasface, eye=None, last_eye=None, last_eye_frame=None, imgsize=None):
        """
        hasface = boolean
        eye = (x,y)
        """
        self.hasface = hasface
        self.eye = eye
        self.last_eye = last_eye
        self.last_eye_frame = last_eye_frame
        self.imgsize = imgsize

        
    def keeping_contact(self):
        """
        Is the person keeping contact?
        """
        if not self.hasface:
            return False
        else:
            print('eye:',self.eye)
            return True

        self.normalized_eye = self.eye / self.imgsize

        if self.normalized_eye:
            pass


opencvpath = '/home/cx1111/Software/opencv/data/haarcascades'
face_cascade = cv2.CascadeClassifier(os.path.join(opencvpath, 'haarcascade_frontalface_default.xml'))
#help(face_cascade.detectMultiScale)
eye_cascade = cv2.CascadeClassifier(os.path.join(opencvpath, 'haarcascade_eye.xml'))


# Video and window
videogen = skvideo.io.vreader('interview.mp4')
app = QtGui.QApplication([])
## Create window with GraphicsView widget
win = pg.GraphicsLayoutWidget()
win.show()  ## show widget alone in its own window
win.setWindowTitle('Conversation Video')
view = win.addViewBox()
## lock the aspect ratio so pixels are always square
view.setAspectLocked(True)
## Create image item
img = pg.ImageItem(border='w')
view.addItem(img)
## Set initial view bounds
#view.setRange(QtCore.QRectF(0, 0, 600, 600))
updateTime = time.time()
fps = 0


# flatframe = cv2.imread('presidents.jpg')
# print(flatframe.shape)
# grayflatframe = cv2.cvtColor(flatframe, cv2.COLOR_BGR2GRAY)
# flatfaces = face_cascade.detectMultiScale(grayflatframe, 1.3, 5)
# for x,y,w,he in flatfaces:
#     flatframe = cv2.rectangle(flatframe,(x,y),(x+w,y+he),(255,0,0),2)
#     pdb.set_trace()
# shape should be height, width, color(480, 720, 3)



fskipcount = 1
fcount = fskipcount


for frame in videogen:
    # Don't need to process every frame
    if fcount < fskipcount:
        fcount += 1
        continue

    # Get the image frame
    grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces and eyes
    faces = face_cascade.detectMultiScale(grayframe, 1.3, 5)

    for x,y,w,he in faces:
        # Draw the face region
        frame = cv2.rectangle(frame,(x,y),(x+w,y+he),(255,0,0),2)

        # Regions of interest
        roi_gray = grayframe[y:y+he, x:x+w]
        roi_color = frame[y:y+he, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)

        # Remove extra eyes
        if len(eyes) > 2:
            eye_sizes = np.array([e[2] for e in eyes])
            largest_two = eye_sizes.argsort()[-2:][::-1]
            eyes = eyes[largest_two]

        for ex,ey,ew,eh in eyes:
            #pdb.set_trace()
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    imgsize = np.array(grayframe.shape[1::-1], dtype='float')
    #print(imgsize)


    # Decide whether keeping contact
    if len(faces) == 0:
        gaze = Gaze(hasface=False)
    else:
        # Choose just one eye
        if len(eyes) == 2:
            # Get the x and y pos of the left eye
            if eyes[0][0] > eyes[1][0]:
                eye = eyes[1][:2]
            else:
                eye = eyes[0][:2]
        else:
            eye = eyes[:2]

        gaze = Gaze(hasface=True, eye=eye, imgsize=imgsize)

    if not gaze.keeping_contact():
        raise_alarm()

    #pdb.set_trace()

    #gaze = Gaze(imsize = np.array(grayframe.shape[:2], dtype='float'))




    # Shift because pyqt is weird
    frame = frame.transpose([1,0,2])[:,-1::-1,:]
    # Display
    img.setImage(frame)
    #time.sleep(0.01)
    app.processEvents()


    # Timer content
    now = time.time()
    fps2 = 1.0 / (now-updateTime)
    updateTime = now
    fps = fps * 0.9 + fps2 * 0.1
    
    #print "%0.1f fps" % fps
    #pdb.set_trace()

    fcount = 0


