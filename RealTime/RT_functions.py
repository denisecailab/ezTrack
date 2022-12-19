"""

LIST OF CLASSES/FUNCTIONS

Video (Class)
hv_baseimage
clear_queues

"""


import os
import sys
import cv2
import pickle
import holoviews as hv
from holoviews import streams
from holoviews.streams import Stream, param
import numpy as np
import time
import datetime
import threading, multiprocessing, queue
import functools as fct
from scipy.ndimage.measurements import center_of_mass
from threading import Thread
from IPython.display import clear_output
hv.notebook_extension('bokeh')



class Video():
    
    """ 
    -------------------------------------------------------------------------------------
    
    Base container for holding video stream, tracking paramaters, and video read/write
    methods.
    
    Of note, data/video saving/writing is completed in separate process 
    (Video.writer_writer).  

    -------------------------------------------------------------------------------------
    
    Methods:
    
        - init
        - scale_set
        - start
        - stop
        - release
        - display
        - get_frames
        - ref_create
        - locate
        - crop_define
        - crop_cropframe
        - mask_define
        - roi_define
        - params_save
        - params_load
        - writer_init
        - writer_start
        - writer_stop
        - writer_writer
    
    Attributes (see __init__ for details):
    
        - stream
        - started
        - frame
        - frame_time
        - ref
        - dif
        - params_loaded
        - crp_bnds
        - mask
        - roi_names
        - roi_masks
        - fps
        - scale
        - scale_orig
        - scale_w
        - scale_h
        - track
        - track_yx
        - track_roi
        - track_thresh
        - track_method
        - track_window_use
        - track_window_sz
        - track_window_wt
        - track_window_reset
        - track_rmvwire
        - track_rmvwire_krn
        - writer_startsig
        - writer_startsig
        - writer_complete
        - q_frm
        - q_frmt
        - q_frmyx
        - q_roi
 
    """

    def __init__(self, color=None, src=0, scale=None, buffer=10):
        
        """ 
        -------------------------------------------------------------------------------------

        Video class initialization

        -------------------------------------------------------------------------------------
        Args:
            src:: [int]
                USB input of camera.
                
            color:: [str]
                Used to select particular channel from color images. Must be set to 'blue',
                'green', 'red', or None. When None, images converted to grayscale.

            scale:: [float, 0<x<=1]
                Downsampling of each frame, such that 0.3 would result in 30% input size. 
                Uses OpenCV INTER_NEAREST method. If no downsampling can be set to None.

            buffer:: [unsigned integer]
                Max size of queues, the buffer used to synchronize frames in main tracking 
                process with data/video writer process.
                
        -------------------------------------------------------------------------------------        
        Attributes:

            stream:: [cv2.VideoCapture]
                OpenCV VideoCapture class instance for video.
                
            color:: [str]
                Used to select particular channel from color images. Must be set to 'blue',
                'green', 'red', or None. When None, images converted to grayscale.

            started:: [boolean]
                Indicates if frames are currently being retrieved. Note that this is distinct
                from Video.track, which indicates whether tracking is ongoing.

            frame:: [array]
                The most recently captured frame. Note that because this is continuously
                updated it is safer to copy than access directly.
                
            frame_time:: [float64]
                Time current frame was acquired by OpenCV.

            ref:: [array]
                Reference frame composed of field of view without animal, necessary for 
                tracking. See Video.ref_create for details.  Same shape as Video.frame.
                
            dif:: [array]
                The features of the most recently captured frame used for tracking.
                Conceptually, dif = frame - ref, but read tracking params below for details.

            params_loaded:: [bool]
                Indicates whether parameters have been loaded from a file using
                Video.params_load method. 

            crp_bnds:: [None, holoviews.streams.BoxEdit or dictionary]
                Defines cropping bounds of frame, after scaling. Set to None if no cropping 
                is to be performed.  Can be drawn in Jupyter Notebook using Video.crop_define,
                subsequently saved using Video.params_save, and then loaded using
                Video.params_load.  To set this manually Video.params_loaded will also need to 
                be set to True.  When defined manualy, Video.crp_bnds should be a dictionary 
                with the following keys: ['x0', 'x1', 'y0', 'y1'], and each value should be a 
                list of length 1.  
                For example "Video.crp_bnds = dict(x0=[5], x1=[500], y0=[10], y1=[300])"

            mask:: [bool array]
                Boolean numpy array identifying regions to exclude from tracking.  Should 
                be same dimensions as Video.frame.

            roi_names:: [list]
                List of region of interest names.

            roi_masks:: [dict]
                Dictionary with Video.roi_names as keys, with corresponding boolean areas of
                shape Video.frame as values.

            fps:: [float]
                Video acquisition rate.

            scale:: [0<x<=1]
                Downsampling of each frame, such that 0.3 would result in 30% 
                input size. Uses OpenCV INTER_NEAREST method. 
                Should be set either when initially defining Video instance, or by using the
                Video.set_scale function.     

            scale_orig:: [tuple]
                Dimensions of original video frame, before downsampling: (width, height).

            scale_w:: [int]
                Width of frame, pre-cropping, including any applied downsampling. 
                Set by Video.scale_set()

            scale_h:: [type]
                Height of frame, pre-cropping, including any applied downsampling.
                Set by Video.scale_set()
                
            track:: [bool]
                Set to True to initiate tracking. Video.started should be True before
                tracking is begun.

            track_yx:: [tuple]
                Indices of center of mass as tuple in the form: (y,x).

            track_roi:: [dictionary]
                Dictionary with Video.roi_names as keys, with corresponding boolean values
                indicating if animal is in each ROI.

            track_thresh:: [float between 0-100]
                Percentile of difference values below which are set to 0. After calculating 
                pixel-wise difference between passed frame and reference frame, these values 
                are thresholded to make subsequent defining of center of mass more reliable. 

            track_method:: [string]
                Set to 'abs', 'light', or 'dark'. If 'abs', absolute difference between 
                reference and current frame is taken, and thus the background of the frame 
                doesn't matter. 'light' specifies that the animal is lighter than the background.
                'dark' specifies that the animal is darker than the background. 

            track_window_use: [bool]
                Will window surrounding prior location be imposed?  Allows changes in area 
                surrounding animal's location on previous frame to be more heavily influential
                in determining animal's current location. After finding pixel-wise difference 
                between passed frame and reference frame, difference values outside square window
                of prior location will be multiplied by (1 - window_weight), reducing their 
                overall influence. [bool]

            track_window_sz:: [unsigned integer]
                If `use_window=True`, the length of one side of square window, in pixels.

            track_window_wt:: [float between 0-1]
                0-1 scale for window, if used, where 1 is maximal weight of window surrounding 
                prior location. 
                
            track_window_reset:: [bool]
                When set to True, window will not be used on tracking of next tracked frame,
                permitting the window position to be reset.

            track_rmvwire:: [bool]
                True/False, indicating whether to use wire removal function. 

            track_rmvwire_krn:: [unsigned integer]
                Size of kernel used for morphological opening to remove wire.
                Should be larger than wire and smaller than animal.
                
            writer_initiated:: [multiprocessing.Event]
                Multiprocessing event used to coordinate with writer process.
                Informs whether writer has been successfully initiated. 
            
            writer_startsig:: [multiprocessing.Event]
                Multiprocessing event used to coordinate with writer process.
                Used to start writing frames after initiation and can be used to
                query state of writer. Safest to invoke with self.writer_start.
            
            writer_stopsig:: [multiprocessing.Event]
                Multiprocessing event used to coordinate with writer process.
                Used to stop writer after initiation and can be used to
                query state of writer. Safest to invoke with vid.stop,
                or vid.release, which will sotp writer and clear queues. 
                Self.writer_stop will similarly stop writing, but will not
                clear queus.
                
            writer_emptyq:: [multiprocessing.Event]
                Multiprocessing event used to coordinate with writer process.
                Can be set just prior to writer_stopsig to ensure that any frames
                remaining in buffer are saved.
                
            writer_complete:: [multiprocessing.Event]
                Multiprocessing event used to coordinate with writer process.
                Signals that writing is complete. 
                
            q_frm:: [multiprocessing.Queue]
                Queue of video frames.  Used for reference creation
                (Video.ref_create) and saving.
                
            q_frmt:: [multiprocessing.Queue]
                Queue of video frame times.  Used for saving.
                
            q_frmyx:: [multiprocessing.Queue]
                Queue of video frame positions.  Used for saving.
                
            q_roi:: [multiprocessing.Queue]
                Queue of ROI state.  Used for saving.

        -------------------------------------------------------------------------------------
        Notes:

        """
        
        self.stream = cv2.VideoCapture(src)
        self.color = color
        self.started = False
        self.frame = None
        self.frame_time = None
        self.ref = None
        self.dif = None       
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
        self.track_window_reset = False
        self.track_rmvwire = False
        self.track_rmvwire_krn = 10
        self.writer_initiated = multiprocessing.Event() 
        self.writer_startsig = multiprocessing.Event() 
        self.writer_stopsig = multiprocessing.Event() 
        self.writer_emptyq = multiprocessing.Event() 
        self.writer_complete = multiprocessing.Event() 
        self.q_frm = multiprocessing.Queue(buffer)
        self.q_frmt = multiprocessing.Queue(buffer)
        self.q_frmyx = multiprocessing.Queue(buffer)
        self.q_roi = multiprocessing.Queue(buffer)

        

    def scale_set(self, scale=1):
        
        """ 
        -------------------------------------------------------------------------------------

        Allows for setting downsampling scale after defining Video instance.  Should be 
        done prior to generating reference from and beginning tracking.

        -------------------------------------------------------------------------------------
        Args:
            scale:: [float, 0<x<=1]
                Downsampling of each frame, such that 0.3 would result in 30% input size. 
                Uses OpenCV INTER_NEAREST method. If no downsampling can be kept = None.

        -------------------------------------------------------------------------------------
        Notes:

        """
        
        self.scale = scale
        self.w = int(self.scale_orig[0]*self.scale) 
        self.h = int(self.scale_orig[1]*self.scale) 

        
        
    def start(self):
        
        """ 
        -------------------------------------------------------------------------------------

        Initializes continuous retrieval of frames in thread. 

        -------------------------------------------------------------------------------------
        Notes:
   
            This does not start tracking (set with Video.track)

        """
        
        self.started = True
        Thread(target=self.get_frames, args=()).start()    
    
    
    
    def stop(self):
        
        """ 
        -------------------------------------------------------------------------------------

        Stops thread in control of continuous retrieval of frames.  

        -------------------------------------------------------------------------------------
        Notes:
        
            Counter to Video.release, this does not release the CV2.VideoCapture instance. 

        """
        
        self.started = False
        if self.writer_initiated.is_set():            
            if self.writer_startsig.is_set():
                self.writer_emptyq.set()
                self.writer_stopsig.set() 
            else:
                self.writer_stopsig.set()
            while not self.writer_complete.is_set():
                time.sleep(.1)
        self.writer_startsig.clear()
        self.writer_stopsig.clear()
        self.writer_complete.clear()
        self.writer_emptyq.clear()
        self.writer_complete.clear()
                
                
        
    def release(self):
        
        """ 
        -------------------------------------------------------------------------------------

        Stops thread in control of continuous retrieval of frames and releases camera.

        -------------------------------------------------------------------------------------
        Notes:
        
            See Video.stop if you would like to stop retrieving frames wihtout releasing 
            camera. 

        """
        
        self.stop()
        self.stream.release()  
  


    def display(self, show_xy=True, show_dif = True):
        
        """ 
        -------------------------------------------------------------------------------------

        OpenCV window allowing display of tracking.  'Q' can be pressed to exit

        -------------------------------------------------------------------------------------
        Args:
            show_xy:: [bool]
                Dictates whether position of animal should be presented, if tracking.  Can
                be safely kept to True when not tracking.
            
            show_dif:: [bool]
                Option to display second window where features used for tracking are
                highlighted.

        -------------------------------------------------------------------------------------
        Notes:

        """
        
        while True:
            frame = self.frame.copy()
            if show_xy==True and self.track_yx is not None:
                markposition = (
                        int(self.track_yx[1]),
                        int(self.track_yx[0]))
                cv2.drawMarker(img=frame,position=markposition,color=255)
            cv2.imshow('Video', frame)
            if show_dif:
                try:
                    cv2.imshow('Difference', self.dif.copy().astype('uint8'))
                except:
                    pass
            #wait for 'q' key response to exit
            if (cv2.waitKey(int(1000/self.fps) & 0xFF) == 113) or not self.started:
                break  
        cv2.destroyAllWindows()
        _=cv2.waitKey(1)

        
        
    def get_frames(self):
        
        """ 
        -------------------------------------------------------------------------------------

        Function executed in thread under Video.start to iteratively retrieve frames and 
        track animal (if Video.track is True).

        -------------------------------------------------------------------------------------
        Notes:

        """
        
        while self.started:
            
            #get latest frame
            try:
                ret, frame = self.stream.read() 
            except:
                ret = False
                
            if ret == True:
                self.frame_time = time.time()
                if self.color is not None:
                    if self.color=='blue':
                        frame = frame[:,:,0]
                    elif self.color=='green':
                        frame = frame[:,:,1]
                    elif self.color=='red':
                        frame = frame[:,:,2]
                    else:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
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
                                int(self.track_yx[0]), int(self.track_yx[1])
                            ]

                #add frame info to video queues
                for q in [self.q_frm, self.q_frmt, self.q_frmyx, self.q_roi]:
                    if q.full():
                        q.get()
                self.q_frm.put(self.frame) 
                self.q_frmt.put(self.frame_time)
                if self.track:
                    self.q_frmyx.put(self.track_yx)
                    self.q_roi.put(self.track_roi)

            
            
    def ref_create(self, print_sts=True, secs=5):
        
        """ 
        -------------------------------------------------------------------------------------

        Generates reference frame from ongoing collection of frames.

        -------------------------------------------------------------------------------------
        Args:
            print_sts:: [bool]
                Option to print progess of reference creation
            secs:: [numeric]
                Number of seconds to generate reference frame from.

        -------------------------------------------------------------------------------------
        Notes:


        """
        
        samples = int(secs*self.fps)
        
        clear_queues([self.q_frm, self.q_frmt, self.q_frmyx, self.q_roi])
        
        for smpl in np.arange(samples):
            frame  = self.q_frm.get()
            if smpl!=0:
                mean = mean*(smpl/(smpl+1)) + frame*(1/(smpl+1))
            else:
                mean = self.frame
            if print_sts:
                print((smpl+1)/samples*secs)
            clear_output(wait=True)
        self.ref = mean
  


    def locate(self, frame):
        
        """ 
        -------------------------------------------------------------------------------------

        Defines position of animal within frame.

        -------------------------------------------------------------------------------------
        Args:
            frame:: [array]
                The frame in which to track the animal.

        -------------------------------------------------------------------------------------
        Notes:


        """
        
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
                print("WARNING: wire_krn too large. Reverting to rmv_wire=False for current frame")
        if self.mask is not None:
            if self.mask['mask'] is not None:
                dif[self.mask['mask']] = 0  
        if self.track_window_use==True and self.track_yx!=None:
            if not self.track_window_reset:
                ymin = int(self.track_yx[0] - (self.track_window_sz//2))
                ymax = int(self.track_yx[0] + (self.track_window_sz//2))
                xmin = int(self.track_yx[1] - (self.track_window_sz//2))
                xmax = int(self.track_yx[1] + (self.track_window_sz//2))
                dif_weights = np.ones(dif.shape)*(1-self.track_window_wt)
                dif_weights[
                    slice(ymin if ymin>0 else 0, ymax),
                    slice(xmin if xmin>0 else 0, xmax)]=1
                dif = dif*dif_weights
            else:
                self.track_window_reset = False
        
        dif[dif<np.percentile(dif,self.track_thresh)]=0
        self.dif = dif.copy()
        com = center_of_mass(self.dif)
        return com
  


    def crop_define(self):
        
        """ 
        -------------------------------------------------------------------------------------

        Holoviews tool allowing one to select cropping bounds in interactive manner using
        box selection tool.

        -------------------------------------------------------------------------------------

        Returns:
            item:: [holoviews.Image]
                holoviews.Image displaying most recent frame.  Use box selection tool to
                select cropping bounds.

        -------------------------------------------------------------------------------------
        Notes:
        
            Can only be used within Jupyter.

        """
        
        image = hv_baseimage(
            frame = self.frame,
            text = "Crop as Desired"
        )
        
        box = hv.Polygons([])
        box.opts(alpha=.5)
        self.crop_bnds = streams.BoxEdit(source=box,num_objects=1)
        return (image*box)
 


    def crop_cropframe(self, frame):
        
        """ 
        -------------------------------------------------------------------------------------

        Crops the passed frame using cropping parameters in Video.crop_bnds

        -------------------------------------------------------------------------------------
        Args:
            frame:: [array]
                Input frame 

        -------------------------------------------------------------------------------------
        Returns:
            frame:: [array]
                Cropped output frame   

        -------------------------------------------------------------------------------------
        Notes:


        """
        
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
        
        """ 
        -------------------------------------------------------------------------------------

        Holoviews tool allowing one to select multiple regions of field of view to be 
        ignored using point selection tool.  Double-click to start/stop region, single-click
        to add vertex to existing region

        -------------------------------------------------------------------------------------
        Returns:
            item:: [holoviews.Image]
                holoviews.Image displaying most recent frame.  Use point selection tool to
                define regions to be ignored.

        -------------------------------------------------------------------------------------
        Notes:
            
            Can only be used within Jupyter.

        """
        
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
        pointDraw_stream = streams.PointDraw(source=points) 
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
        
        """ 
        -------------------------------------------------------------------------------------

        Holoviews tool allowing one to select regions of interest using point selection tool.  
        Double-click to start/stop region, single-click to add vertex to existing region.

        -------------------------------------------------------------------------------------
        Args:
            names:: [list]
                Names of regions of interest.
                
        -------------------------------------------------------------------------------------
        Returns:
            item:: [holoviews.Image]
                holoviews.Image displaying most recent frame.  Use point selection tool to
                define regions of interest.

        -------------------------------------------------------------------------------------
        Notes:

            Can only be used within Jupyter.

        """
    
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
        
        """ 
        -------------------------------------------------------------------------------------

        Saves Video.roi_masks, Video.crop_bnds, Video.mask, and Video.scale within a pickle
        file that can be subsequently loaded and run using Video.params_load.
        
        -------------------------------------------------------------------------------------
        Args:
            file:: [string]
                Filename of item to be saved.  Must have '.pickle' extension. 

        -------------------------------------------------------------------------------------
        Notes:


        """
        
        vid_dict = {
            'roi_masks' : self.roi_masks,
            'crop_bnds' : None if self.crop_bnds is None else self.crop_bnds.data,
            'mask' : None if self.mask is None else self.mask['mask'],
            'scale' : self.scale
        }
        with open(file, 'wb') as pickle_file:
            pickle.dump(vid_dict, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
    
    
    
    def params_load(self, file=None):
        
        """ 
        -------------------------------------------------------------------------------------

        Loads Video.roi_masks, Video.crop_bnds, Video.mask, and Video.scale from a pickle
        file and adds them to Video object. 
        
        -------------------------------------------------------------------------------------
        Args:
            file:: [string]
                Filename of item to be saved.  Must have '.pickle' extension. 

        -------------------------------------------------------------------------------------
        Notes:


        """
        
        with open(file, 'rb') as pickle_file:
            vid_dict = pickle.load(pickle_file)   
        self.roi_masks = vid_dict['roi_masks']
        self.roi_names = None if self.roi_masks is None else list(self.roi_masks.keys())
        self.track_roi = {x : None for x in self.roi_names} if self.roi_names is not None else None
        self.crop_bnds = vid_dict['crop_bnds']
        self.mask = {'mask' : vid_dict['mask']}
        self.scale = vid_dict['scale']
        if self.scale != 1:
            self.scale_set(scale = self.scale)
        self.params_loaded = True


        
    def writer_init(
        self,
        save_vid = True,
        save_csv = True,
        dpath = './',
        dfilename = None,
        vfilename = None, 
        compression = "MJPG", 
        fps=30
    ):
        
        """ 
        -------------------------------------------------------------------------------------

        Creates cv2.VideoWriter process and empty csv to save data to.
        
        -------------------------------------------------------------------------------------
        Args:
            save_vid:: [bool]
                Whether or not to save video.
                
            save_csv:: [bool]
                Whether or not to save csv.
                
            dpath:: [string]
                Directory to save data/video file to.
                
            dfilename:: [string]
                Name of file to save to.  Must have '.csv' extension. 
                If None, file will be set to YearMonthDay_HourMinuteSecond.csv
                
            vfilename:: [string]
                Name of video file to save to.  Must have '.avi' extension
                If None, file will be set to YearMonthDay_HourMinuteSecond.avi
                
            compression:: [string]
                Video compression algorithm.  Choose from the following:
                    0 = no compression
                    'MJPG' = lossly compression.
                    'FFV1' = lossless compression
                    'XVID' = lossless compression
            
            fps:: [int]
                FPS written to codec.  May only except certain values.

        -------------------------------------------------------------------------------------
        Notes:
            
            Does not begin saving.  Only creates writer object.
            Use Video.writer_start and Video.writer_stop to initiate/stop saving.


        """
        
        t = datetime.datetime.fromtimestamp(time.time()).strftime("%Y%M%d_%H%M%S")
        if dfilename is None:
            dfilename = '.'.join([t, '.csv'])
        if vfilename is None:
            vfilename = '.'.join([t, '.avi'])
        cpath = os.path.join(os.path.abspath(dpath), dfilename)
        vpath = os.path.join(os.path.abspath(dpath), vfilename)
    
        multiprocessing.Process(
            target=self.writer_writer,
            args=(
                save_vid,
                save_csv,
                cpath,
                vpath,
                compression,
                fps,
                self.roi_names,
                self.q_frm,
                self.q_frmt,
                self.q_frmyx,
                self.q_roi,
                self.frame.copy().shape[1], 
                self.frame.copy().shape[0],
                self.writer_initiated,
                self.writer_startsig,
                self.writer_stopsig,
                self.writer_emptyq,
                self.writer_complete,
                self.track
            )
        ).start()
        while not self.writer_initiated.is_set():
            pass

        
        
    def writer_start(self):
        
        """ 
        -------------------------------------------------------------------------------------

        Signals to begin writing frames.  

        -------------------------------------------------------------------------------------       
        Notes:
        
            Must follow vid.writer_init()

        """
        
        self.writer_startsig.set()


        
    def writer_stop(self):
        
        """ 
        -------------------------------------------------------------------------------------

        Signals to end writing frames.  

        -------------------------------------------------------------------------------------       
        Notes:
        
            Must follow vid.writer_init()

        """
        
        self.writer_stopsig.set()


        
    @staticmethod
    def writer_writer(
        save_vid, save_csv, cpath, vpath, compression, fps, roi_names, 
        q_frm, q_frmt, q_frmyx, q_roi, scale_w, scale_h, 
        initsig, startsig, stopsig, emptyq, complete, track
    ):
        
        """ 
        -------------------------------------------------------------------------------------

        Creates cv2.VideoWriter object.  By default executed by vid.writer_init as distinct
        process.
        
        -------------------------------------------------------------------------------------
        Args:
            save_vid:: [bool]
                Whether or not to save video.
                
            save_csv:: [bool]
                Whether or not to save csv.
                
            cpath:: [string]
                Full path of csv file to save data to.
            
            vpath:: [string]
                Full path of file to save video file to.
                
            compression:: [string]
                Compression algorithm.  Choose from the following:
                    0 = no compression
                    'MJPG' = lossly compression.
                    'FFV1' = lossless compression
                    'XVID' = lossless compression
            
            fps:: [int]
                FPS written to codec.  May only except certain values.
                
            roi_names:: [list]
                List of region of interest names.
                
            q_frm:: [multiprocessing.Queue]
                Queue of video frames.  Used for reference creation
                (Video.ref_create) and saving.
                
            q_frmt:: [multiprocessing.Queue]
                Queue of video frame times.  Used for saving.
                
            q_frmyx:: [multiprocessing.Queue]
                Queue of video frame positions.  Used for saving.
                
            q_roi:: [multiprocessing.Queue]
                Queue of ROI state.  Used for saving.
                
            scale_w:: [int]
                Width of frame.
                
            scale_h:: [int]
                Height of frame
                
            initsig:: [multiprocessing.Event]
                Multiprocessing event used to coordinate with writer process.
                Informs whether writer has been successfully initiated. 
                
            startsig:: [multiprocessing.Event]
                Multiprocessing event used to coordinate with main process.
                Used to start writing frames after initiation and can be used to
                query state of writer. Safest to invoke with self.writer_start.
                
            stopsig:: [multiprocessing.Event]
                Multiprocessing event used to coordinate with main process.
                Used to stop writer after initiation and can be used to
                query state of writer. Safest to invoke with vid.stop,
                or vid.release, which will sotp writer and clear queues. 
                Self.writer_stop will similarly stop writing, but will not
                clear queus.
                
            emptyq:: [multiprocessing.Event]
                Multiprocessing event used to coordinate with writer process.
                Can be set just prior to writer_stopsig to ensure that any frames
                remaining in buffer are saved.
            
            complete:: [multiprocessing.Event]
                Multiprocessing event used to coordinate with main process.
                 Signals that writing is complete.
            
            track:: [bool]
                Boolean indicating whether tracking data is to be saved.

               

        -------------------------------------------------------------------------------------
        Notes:

        """
        
        import pandas as pd
        import numpy as np
        import cv2
        import time
        
        #initialize video and csv
        dkeys = ['frame_time']
        if track:
            dkeys = dkeys + ['y', 'x']
            dkeys = dkeys if roi_names is None else dkeys + roi_names
        if save_vid:
            writer = cv2.VideoWriter(
                filename = vpath, 
                fourcc = cv2.VideoWriter_fourcc(*compression) if compression in ["MJPG", "FFV1", "XVID"] else 0, 
                fps = fps, 
                frameSize = (scale_w, scale_h),
                isColor = False
            )
        if save_csv:
            pd.DataFrame(columns = dkeys).to_csv(cpath, header=True, index=False)
        initsig.set()
        
        #await start (or stop) signal
        while not startsig.is_set():
            time.sleep(.01)            
            if stopsig.is_set():
                break
        clear_queues([q_frm, q_frmt, q_frmyx, q_roi])         
        
        #write data
        while not stopsig.is_set() or emptyq.is_set():
            try:
                dcsv = dict()
                frame = q_frm.get(timeout=1/fps)
                dcsv['frame_time'] = q_frmt.get(timeout=1/fps)
                if track:
                    frame_yx = q_frmyx.get(timeout=1/fps)  
                    roi = q_roi.get(timeout=1/fps) if roi_names is not None else None
                if save_csv:
                    dcsv['y'], dcsv['x'] = frame_yx if track else (None, None)
                    dcsv.update(roi) if (track and roi_names is not None) else None
                    pd.DataFrame(
                        {k:v for k,v in dcsv.items() if k in dkeys},
                        index=[0]
                    ).to_csv(cpath, header=False, index=False, mode='a')
                if save_vid:
                    writer.write(frame)                     
            except:
                if emptyq.is_set():
                    break
                pass
        
        #close out
        time.sleep(.1)
        if save_vid:
            writer.release() 
        complete.set()

        

def hv_baseimage(frame, text=None):  
    
    """ 
    -------------------------------------------------------------------------------------

    Base holoviews image to subsequently manipulate

    -------------------------------------------------------------------------------------
    Args:
        frame:: [array]
            Frame to use.

    -------------------------------------------------------------------------------------
    Returns:
        image:: [hv.Image]
            holoviews.Image which can then be modified/added to.  

    -------------------------------------------------------------------------------------
    Notes:


    """
        
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

    

def clear_queues(queues):
    
    """ 
    -------------------------------------------------------------------------------------

    Used to empty multiprocessing queues.

    -------------------------------------------------------------------------------------
    Args:
        queues:: [list]
            List containing multiprocessing queues to be cleared

    -------------------------------------------------------------------------------------
    Notes:


    """
    
    while any([not q.empty() for q in queues]):
        for q in queues:
            try:
                _ = q.get_nowait()
            except:
                pass


   

        
        
        
        
        
        
   
    

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
