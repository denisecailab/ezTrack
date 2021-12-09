import os
import sys
import cv2
import h5py
import pickle
import holoviews as hv
from holoviews import streams
from holoviews.streams import Stream, param
import numpy as np
import time
import threading, multiprocessing, queue
import functools as fct
from scipy.ndimage.measurements import center_of_mass
from threading import Thread
from IPython.display import clear_output
hv.notebook_extension('bokeh')



class Video():

    
    def __init__(self, src=0, scale=None, buffer=10):
        self.stream = cv2.VideoCapture(src)
        self.started = False
        self.frame = None
        self.ref = None
        self.fq = queue.Queue(buffer)
        self.params_loaded = False
        self.crop_bnds = None
        self.mask = None
        self.roi_names = None
        self.roi_masks = None
        self.fps = self.stream.get(cv2.CAP_PROP_FPS)
        self.scale = scale if scale is not None else 1
        self.scale_orig = (self.stream.get(cv2.CAP_PROP_FRAME_WIDTH), self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
        self.scale_w = int(self.scale_orig[0]*self.scale) 
        self.scale_h = int(self.scale_orig[1]*self.scale) 
        self.track = False
        self.track_yx = None
        self.track_roi = None
        self.track_thresh = 99
        self.track_method = 'abs'
        self.track_window_use = False
        self.track_window_sz = 100
        self.track_window_wt = 0.9
        self.track_rmvwire = False
        self.track_rmvwire_krn = 10
  


    def scale_set(self, scale=1):
        self.scale = scale
        self.w = int(self.scale_orig[0]*self.scale) 
        self.h = int(self.scale_orig[1]*self.scale) 

        
        
    def start(self): 
        self.started = True
        Thread(target=self.get_frames, args=()).start()    
    
    
    
    def stop(self):
        self.started = False
        
        
        
    def release(self):
        self.started = False
        time.sleep(.1)
        self.stream.release()  
  


    def display(self, display=True, show_xy=False):
        while display==True:
            frame = self.frame.copy()
            if show_xy==True and self.track_yx is not None:
                markposition = (
                        int(self.track_yx[1]),
                        int(self.track_yx[0]))
                cv2.drawMarker(img=frame,position=markposition,color=255)
            cv2.imshow('Video', frame)
            #wait for 'q' key response to exit
            if (cv2.waitKey(int(1000/self.fps) & 0xFF) == 113):
                display = False       
        cv2.destroyAllWindows()
        _=cv2.waitKey(1)

        
        
    def get_frames(self):
        while self.started:
            
            #get latest frame
            ret, frame = self.stream.read() 
            if ret == True:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                #scale, crop, and mask frame
                if self.scale != 1:
                    frame = cv2.resize(
                        frame,
                        (self.scale_w, self.scale_h),
                        cv2.INTER_NEAREST)
                if self.crop_bnds is not None:
                    frame = self.crop_cropframe(frame)
                self.frame = frame

                #track locatiotn
                if self.track:
                    self.track_yx = self.locate(self.frame)
                    if self.roi_masks is not None:
                        for roi in self.roi_masks.keys():
                            self.track_roi[roi] = self.roi_masks[roi][
                                int(self.track_yx[0]), int(self.track_yx[1])]

                #add frame to video queue
                if self.fq.full():
                    self.fq.queue.popleft()
                self.fq.put(self.frame)             

            
            
    def ref_create(self, secs=5):
        samples = int(secs*self.fps)
        self.fq.queue.clear()
        for smpl in np.arange(samples):
            frame  = self.fq.get()
            if smpl!=0:
                mean = mean*(smpl/(smpl+1)) + frame*(1/(smpl+1))
            else:
                mean =self.frame
            print((smpl+1)/samples*secs)
            clear_output(wait=True)
        self.ref = mean
  


    def locate(self,frame):
        if self.track_method == 'abs':
            dif = np.absolute(frame-self.ref)
        elif self.track_method == 'light':
            dif = frame-self.ref
        elif self.track_method == 'dark':
            dif = self.ref-frame
        dif = (dif - dif.min()).astype('int16') #scale so lowest value is 0
        
        if self.track_rmvwire == True:
            kernel = np.ones((self.track_rmvwire_krn, self.track_rmvwire_krn),np.uint8)
            dif_wirermv = cv2.morphologyEx(dif, cv2.MORPH_OPEN, kernel)
            krn_violation =  dif_wirermv.sum()==0
            dif = dif if krn_violation else dif_wirermv
            if krn_violation:
                print("WARNING: wire_krn too large. Reverting to rmv_wire=False for frame {x}".format(
                    x= int(cap.get(cv2.CAP_PROP_POS_FRAMES)-1-video_dict['start'])))
        if self.mask is not None:
            if self.mask['mask'] is not None:
                dif[self.mask['mask']] = 0  
        if self.track_window_use==True and self.track_yx!=None:
            ymin = int(self.track_yx[0] - (self.track_window_sz//2))
            ymax = int(self.track_yx[0] + (self.track_window_sz//2))
            xmin = int(self.track_yx[1] - (self.track_window_sz//2))
            xmax = int(self.track_yx[1] + (self.track_window_sz//2))
            dif_weights = np.ones(dif.shape)*(1-self.track_window_wt)
            dif_weights[
                slice(ymin if ymin>0 else 0, ymax),
                slice(xmin if xmin>0 else 0, xmax)]=1
            dif = dif*dif_weights
        
        dif[dif<np.percentile(dif,self.track_thresh)]=0
        com = center_of_mass(dif)
        return com
  


    def crop_define(self):
        image = hv_baseimage(
            frame = self.frame,
            text = "Crop as Desired"
        )
        
        box = hv.Polygons([])
        box.opts(alpha=.5)
        self.crop_bnds = streams.BoxEdit(source=box,num_objects=1)
        return (image*box)
 


    def crop_cropframe(self, frame):   
        if self.params_loaded:
            try:
                Xs = [self.crop_bnds['x0'][0], self.crop_bnds['x1'][0]]
                Ys = [self.crop_bnds['y0'][0], self.crop_bnds['y1'][0]]
                fxmin,fxmax=int(min(Xs)), int(max(Xs))
                fymin,fymax=int(min(Ys)), int(max(Ys))
                return frame[fymin:fymax,fxmin:fxmax]
            except:
                return frame  
        else:
            try:
                Xs = [self.crop_bnds.data['x0'][0], self.crop_bnds.data['x1'][0]]
                Ys = [self.crop_bnds.data['y0'][0], self.crop_bnds.data['y1'][0]]
                fxmin,fxmax=int(min(Xs)), int(max(Xs))
                fymin,fymax=int(min(Ys)), int(max(Ys))
                return frame[fymin:fymax,fxmin:fxmax]
            except:
                return frame  
 


    def mask_define(self):
        
        #Make base image on which to draw
        image = hv_baseimage(
            frame = self.frame,
            text = "Draw Regions to be Excluded"
        )

        #Create polygon element on which to draw and connect via stream to PolyDraw drawing tool
        self.mask = dict(mask=None)
        poly = hv.Polygons([])
        self.mask['stream'] = streams.PolyDraw(source=poly, drag=True, show_vertices=True)
        poly.opts(fill_alpha=0.3, active_tools=['poly_draw'])
        points = hv.Points([]).opts(active_tools=['point_draw'], color='red',size=10)
        pointDraw_stream = streams.PointDraw(source=points,num_objects=20) 
        def make_mask(data, mask):
            try:
                x_ls, y_ls = data['xs'], data['ys'] 
            except TypeError:
                x_ls, y_ls = [], []
            if len(x_ls)>0:
                mask['mask'] = np.zeros(self.frame.shape) 
                for submask in range(len(x_ls)):
                    x = np.array(mask['stream'].data['xs'][submask]) #x coordinates
                    y = np.array(mask['stream'].data['ys'][submask]) #y coordinates
                    xy = np.column_stack((x,y)).astype('uint64') #xy coordinate pairs
                    cv2.fillPoly(mask['mask'], pts =[xy], color=1) #fill polygon  
                    mask['mask'] = mask['mask'].astype('bool')
            return hv.Labels((0,0,""))
        make_mask_ptl = fct.partial(make_mask, mask=self.mask)  
        dmap = hv.DynamicMap(make_mask_ptl, streams=[self.mask['stream']])    
        return image*poly*dmap

    
    
    def roi_define(self, names = None):
    
        self.roi_names = names
        self.track_roi = {x : None for x in self.roi_names}
        self.roi_masks = {} if self.roi_names is not None else None
        nobjects = len(self.roi_names) if self.roi_names is not None else 0 

        #Make base image on which to draw
        image = hv_baseimage(
            frame = self.frame,
            text = "No Regions to Draw" if nobjects == 0 else "Draw Regions: "+', '.join(self.roi_names)
        )

        #Create polygon element on which to draw and connect via stream to PolyDraw drawing tool
        poly = hv.Polygons([])
        self.roi_stream = streams.PolyDraw(source=poly, drag=True, num_objects=nobjects, show_vertices=True)
        poly.opts(fill_alpha=0.3, active_tools=['poly_draw'])
        def centers(data):
            try:
                x_ls, y_ls = data['xs'], data['ys']
            except TypeError:
                x_ls, y_ls = [], []
            xs = [np.mean(x) for x in x_ls]
            ys = [np.mean(y) for y in y_ls]
            rois = self.roi_names[:len(xs)]
            for poly in range(len(xs)):
                x = np.array(self.roi_stream.data['xs'][poly]) 
                y = np.array(self.roi_stream.data['ys'][poly]) 
                xy = np.column_stack((x,y)).astype('uint64') 
                mask = np.zeros(self.frame.shape) 
                cv2.fillPoly(mask, pts =[xy], color=255) 
                self.roi_masks[self.roi_names[poly]] = mask==255 
            return hv.Labels((xs, ys, rois))
        if nobjects > 0:
            dmap = hv.DynamicMap(centers, streams=[self.roi_stream])
            return image * poly * dmap
        else:
            return image
        
    
    
    def params_save(self, file=None):     
        vid_dict = {
            'roi_masks' : self.roi_masks,
            'crop_bnds' : None if self.crop_bnds is None else self.crop_bnds.data,
            'mask' : None if self.mask is None else self.mask['mask'],
            'scale' : self.scale
        }
        with open(file, 'wb') as pickle_file:
            pickle.dump(vid_dict, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
    
    
    
    def params_load(self, file=None):
        with open(file, 'rb') as pickle_file:
            vid_dict = pickle.load(pickle_file)   
        self.roi_masks = vid_dict['roi_masks']
        self.roi_names = None if self.roi_masks is None else list(self.roi_masks.keys())
        self.crop_bnds = vid_dict['crop_bnds']
        self.mask = {'mask' : vid_dict['mask']}
        self.scale = vid_dict['scale']
        if self.scale != 1:
            self.scale_set(scale = self.scale)
        self.params_loaded = True

        

def hv_baseimage(frame, text=None):
    image = hv.Image((
            np.arange(frame.shape[1]),
            np.arange(frame.shape[0]),
            frame))
    image.opts(
        width=int(frame.shape[1]),
        height=int(frame.shape[0]),
        invert_yaxis=True,
        cmap='gray',
        colorbar=True,
        toolbar='below',
        title=text)
    return image



#
# This has not been tested in quite a while
# Not sure if it works
#
class Saver():

    def __init__(self, vid, dpath, bufsize=300):
        self.vpath = os.path.join(dpath,'vid.hd5f')
        self.scale = vid.scale
        self.fq = vid.fq
        self.bufsize = bufsize
        self.buffer = np.zeros(
            (self.scale.h,self.scale.w,self.bufsize)
            ).astype('uint8')
        self.bufqueue = multiprocessing.Queue()
        self.vidstartlen = self.bufsize*10
        self.createfile()
        self.started = False
        self.stopsig = multiprocessing.Event()

    def start(self):
        self.started = True
        self.fq.queue.clear()
        multiprocessing.Process(
            target=self.savebuffer, args=(
                self.vpath, 
                self.bufqueue,
                self.vidstartlen, 
                self.stopsig)).start()
        Thread(target=self.fillbuffer, args=()).start() 
        
    def stop(self):
        self.started = False

    def fillbuffer(self):
        buffer_idx = 0
        f = 0
        while self.started:
            frame = self.fq.get()
            f += 1
            self.buffer[:,:,buffer_idx]=frame
            buffer_idx += 1
            if buffer_idx == self.bufsize:
                self.bufqueue.put(self.buffer)
                buffer_idx = 0
        if buffer_idx != 0:
            self.bufqueue.put(self.buffer[:,:,0:buffer_idx])
        time.sleep(.1)
        self.stopsig.set()
        print('buffer closed')
        print('frames got: {x}'.format(x=f))

    def createfile(self):
        with h5py.File(name = self.vpath, mode = 'w') as vfile:
            vfile.create_dataset(
                name = 'video',
                shape = (self.scale.h, self.scale.w, self.vidstartlen),
                maxshape = (self.scale.h, self.scale.w, None),
                compression="lzf",
                dtype = 'i1',
                chunks = (self.scale.h, self.scale.w, self.bufsize))

    @staticmethod
    def savebuffer(vpath, bufqueue, filelen, stopsig):
        fnums=[0,0]
        with h5py.File(name = vpath, mode = 'r+') as vfile:
            video = vfile['video']
            while not stopsig.is_set():
                try:
                    buffer = bufqueue.get(timeout=10)
                    fnums[0] = fnums[1]
                    fnums[1] = fnums[0] + buffer.shape[2]
                    if fnums[1] >= filelen:
                        filelen += 1000
                        video.resize(
                            (buffer.shape[0], buffer.shape[1], filelen))
                    video[:,:,fnums[0]:fnums[1]] = buffer
                except:
                    None
            video.resize((buffer.shape[0], buffer.shape[1], fnums[1]))
        print('frames saved: {x}'.format(x=fnums[1])) 
        stopsig.clear()      



