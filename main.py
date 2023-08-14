import numpy as np
import cv2
import math
def dirtovec(dir):
    cosin=math.cos(dir[1])
    vec=np.array([cosin*math.cos(dir[0]),cosin*math.sin(dir[0]),math.sin(dir[1])],float)
    return vec
def dirtoscreen(dir,fow,resolution):
    tg=math.tan(fow/2)
    left=np.array([-1*math.sin(dir[0]),math.cos(dir[0]),0],float)*tg
    misin=-math.sin(dir[1])
    up=np.array([misin*math.cos(dir[0]),misin*math.sin(dir[0]),math.cos(dir[1])],float)*tg*resolution[1]/resolution[0]
    screen=np.zeros((resolution[0],resolution[1],3),float)
    screenx = np.zeros((resolution[0],  3), float)
    screenx[:,0]=np.linspace(left[0],-left[0],num=resolution[0])
    screenx[:, 1]= np.linspace(left[1], -left[1], num=resolution[0])
    screenx[:, 2]= np.linspace(left[2], -left[2], num=resolution[0])
    screeny = np.zeros((resolution[1],  3), float)
    screeny[:,0]=np.linspace(up[0],-up[0],num=resolution[1])
    screeny[:, 1]= np.linspace(up[1], -up[1], num=resolution[1])
    screeny[:, 2]= np.linspace(up[2], -up[2], num=resolution[1])
    screen+=screenx[:,np.newaxis,:]
    screen += screeny[ np.newaxis, :,:]
    screen+=dirtovec(dir)[np.newaxis,np.newaxis,:]
    screen/=(np.sum(screen*screen,axis=2)**0.5)[:,:,np.newaxis]
    return screen
class Camera:
    def __init__(self,coord=np.array([0,0,0],float),direction=np.array([0,0],float),fow=103.0/180.0*math.pi,resolution=np.array([1920,1080],int)):
        self.resolution=resolution
        self.coord=coord
        self.direction=direction
        self.fow=fow
        self.screen=dirtoscreen(direction,fow,resolution)
    def changecoord(self,a):
        self.coord=a
    def changedirection(self,a):
        self.direction=a
    def intersect(self,objects):
        resolution=self.resolution
        pixel=np.empty((resolution[0],resolution[1]))
        pixel[:]=np.nan
        for i in objects:
            intersect=i.intersect(self.coord,self.screen,resolution)
            pixel=np.where((np.isnan(pixel) | (intersect<pixel)),intersect,pixel)
        return pixel
class Object:
    def __init__(self):
        pass
    def print(self):
        print(self.shape)
class Sphere(Object):
    def __init__(self,coord,r):
        self.shape='Sphere'
        self.coord = coord
        self.r = r
    def intersect(self,rcoord,rscreen,resolution):
        b=2*np.dot(rscreen,rcoord-self.coord)
        c=np.linalg.norm(rcoord-self.coord)**2-self.r**2
        delta=b**2-4*c[np.newaxis,np.newaxis]
        sphereintersect = np.zeros((resolution[0], resolution[1]), float)
        delta=np.where(delta<0,0,delta)
        t1=np.where(delta > 0, (-b + np.sqrt(delta)) / 2, np.nan)
        t1[t1<0]=np.nan
        t2=np.where(delta > 0, (-b - np.sqrt(delta)) / 2, np.nan)
        t2[t2 < 0] = np.nan
        t1[-1,-1]=np.nan
        print(np.shape(t1))
        t=np.stack((t1, t2))
        t=np.where(np.isnan(t), t[::-1,:,:],t)
        sphereintersect=np.min(t,axis=0)
        return sphereintersect



resolution=np.array([1920,1080],int)
canvas_width = resolution[0]
canvas_height = resolution[1]

a=Camera(resolution=resolution)
b1=Sphere(np.array([2,0,0.5],float),1.0)
b2=Sphere(np.array([2,0.5,-0.5],float),1.0)
z = np.transpose(np.isnan(a.intersect([b1,b2])))
print(z)
while(True):
    blank_image = np.zeros((resolution[1], resolution[0], 3), np.uint8)
    blank_image[z == False] = (255, 255, 255)
    cv2.imshow("blank image",blank_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()



