import cv2
import numpy as np

def bbox(points):
    """
    [xmin xmax]
    [ymin ymax]
    """
    a = np.zeros((2,2))
    a[:,0] = np.min(points, axis=0)
    a[:,1] = np.max(points, axis=0)
    return a

Img=np.zeros((128, 128,3), dtype=np.uint8)

list=[[23,34],[25,67],[64,38],[109,78],[59,49],[108,10],[19,120]]
im2 = np.array(list)
x=im2[0:3,:]
x=np.array(x)

x.
print()
#print(im2.shape[0])

Img[im2[:, 0], im2[:, 1]]=[255,255,255]

cnt = im2
#print(bbox(cnt))
y1 = int(bbox(cnt).item(0))
x1 = int(bbox(cnt).item(2))

y2 = int(bbox(cnt).item(1))
x2 = int(bbox(cnt).item(3))

dist = np.sqrt( (x2 - x1)**2 + (y2 - y1)**2)
#print(dist)
cv2.rectangle(Img,(x1,y1),(x2,y2),(0,255,0),2)


