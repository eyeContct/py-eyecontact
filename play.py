# Reading in frame by frame
import pdb
import skvideo.io
import skvideo.datasets
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
#from pyqtgraph.ptime import time as ptime
import time

videogen = skvideo.io.vreader('examplevideo.mp4')

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

for frame in videogen:
    frame = frame.transpose([1,0,2])[:,-1::-1,:]
    img.setImage(frame)
    #time.sleep(0.01)
    app.processEvents()

    now = time.time()
    fps2 = 1.0 / (now-updateTime)
    updateTime = now
    fps = fps * 0.9 + fps2 * 0.1
    
    print "%0.1f fps" % fps