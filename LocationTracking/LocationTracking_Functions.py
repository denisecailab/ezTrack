"""

LIST OF FUNCTIONS

LoadAndCrop
cropframe
Reference
Locate
TrackLocation
LocationThresh_View
ROI_plot
ROI_Location
Batch_LoadFiles
Batch_Process
PlayVideo
PlayVideo_ext
showtrace
Heatmap
DistanceTool
ScaleDistance

"""





########################################################################################

import os
import sys
import cv2
import fnmatch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import PIL.Image
import time
import warnings
import functools as fct
from scipy import ndimage
from tqdm import tqdm
import holoviews as hv
from holoviews import opts
from holoviews import streams
from holoviews.streams import Stream, param
from io import BytesIO
from IPython.display import clear_output, Image, display
hv.notebook_extension('bokeh')
warnings.filterwarnings("ignore")





########################################################################################    

def LoadAndCrop(video_dict,cropmethod=None,fstfile=False,accept_p_frames=False):
    """ 
    -------------------------------------------------------------------------------------
    
    Loads video and creates interactive cropping tool (video_dict['crop'] from first frame. In the 
    case of batch processing, the first frame of the first video is used. Additionally, 
    when batch processing, the same cropping parameters will be appplied to every video.  
    Care should therefore be taken that the region of interest is in the same position across 
    videos.
    
    -------------------------------------------------------------------------------------
    Args:
        video_dict:: [dict]
            Dictionary with the following keys:
                'dpath' : directory containing files [str]
                'file' : filename with extension, e.g. 'myvideo.wmv' [str]
                'start' : frame at which to start. 0-based [int]
                'end' : frame at which to end.  set to None if processing 
                        whole video [int]
                'region_names' : list of names of regions.  if no regions, set to None
                'dsmpl' : proptional degree to which video should be downsampled
                        by (0-1).
                'stretch' : Dictionary used to alter display of frames, with the following keys:
                        'width' : proportion by which to stretch frame width [float]
                        'height' : proportion by which to stretch frame height [float]
                        *Does not influence actual processing, unlike dsmpl.
                'reference': Reference image that the current frame is compared to. [numpy.array]
                'roi_stream' : Holoviews stream object enabling dynamic selection in response to 
                               selection tool. `poly_stream.data` contains x and y coordinates of roi 
                               vertices. [hv.streams.stream]
                'crop' : Enables dynamic box selection for selection of cropping parameters
                         [hv.streams.BoxEdit]
                'mask' : [dict]
                    Dictionary with the following keys:
                        'mask' : boolean numpy array identifying regions to exlude
                                 from analysis.  If no such regions, equal to
                                 None. [bool numpy array)   
                        'mask_stream' : Holoviews stream object enabling dynamic selection 
                                in response to selection tool. `mask_stream.data` contains 
                                x and y coordinates of region vertices. [holoviews polystream]
                'scale:: [dict]
                        Dictionary with the following keys:
                            'px_distance' : distance between reference points, in pixels [numeric]
                            'true_distance' : distance between reference points, in desired scale 
                                               (e.g. cm) [numeric]
                            'true_scale' : string containing name of scale (e.g. 'cm') [str]
                            'factor' : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
                'ftype' : (only if batch processing) 
                          video file type extension (e.g. 'wmv') [str]
                'FileNames' : (only if batch processing)
                              List of filenames of videos in folder to be batch 
                              processed.  [list]
                'f0' : (only if batch processing)
                        first frame of video [numpy array]
                
        cropmethod:: [str]
            Method of cropping video.  cropmethod takes the following values:
                None : No cropping 
                'Box' : Create box selection tool for cropping video
                
        fstfile:: [bool]
            Dictates whether to use first file in video_dict['FileNames'] to generate
            reference.  True/False
        
        accept_p_frames::[bool]
            Dictates whether to allow videos with temporal compresssion.  Currenntly, if
            more than 1/100 frames returns false, error is flagged.
    
    -------------------------------------------------------------------------------------
    Returns:
        image:: [holoviews.Image]
            Holoviews hv.Image displaying first frame
            
        video_dict:: [dict]
            Dictionary with the following keys:
                'dpath' : directory containing files [str]
                'file' : filename with extension, e.g. 'myvideo.wmv' [str]
                'start' : frame at which to start. 0-based [int]
                'end' : frame at which to end.  set to None if processing 
                        whole video [int]
                'region_names' : list of names of regions.  if no regions, set to None
                'dsmpl' : proptional degree to which video should be downsampled
                        by (0-1).
                'stretch' : Dictionary used to alter display of frames, with the following keys:
                        'width' : proportion by which to stretch frame width [float]
                        'height' : proportion by which to stretch frame height [float]
                        *Does not influence actual processing, unlike dsmpl.
                'reference': Reference image that the current frame is compared to. [numpy.array]
                'roi_stream' : Holoviews stream object enabling dynamic selection in response to 
                               selection tool. `poly_stream.data` contains x and y coordinates of roi 
                               vertices. [hv.streams.stream]
                'crop' : Enables dynamic box selection for selection of cropping parameters
                         [hv.streams.BoxEdit]
                'mask' : [dict]
                    Dictionary with the following keys:
                        'mask' : boolean numpy array identifying regions to exlude
                                 from analysis.  If no such regions, equal to
                                 None. [bool numpy array)   
                        'mask_stream' : Holoviews stream object enabling dynamic selection 
                                in response to selection tool. `mask_stream.data` contains 
                                x and y coordinates of region vertices. [holoviews polystream]
                'scale:: [dict]
                        Dictionary with the following keys:
                            'px_distance' : distance between reference points, in pixels [numeric]
                            'true_distance' : distance between reference points, in desired scale 
                                               (e.g. cm) [numeric]
                            'true_scale' : string containing name of scale (e.g. 'cm') [str]
                            'factor' : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
                'ftype' : (only if batch processing) 
                          video file type extension (e.g. 'wmv') [str]
                'FileNames' : (only if batch processing)
                              List of filenames of videos in folder to be batch 
                              processed.  [list]
                'f0' : (only if batch processing)
                        first frame of video [numpy array]
                
    
    -------------------------------------------------------------------------------------
    Notes:
        - in the case of batch processing, video_dict['file'] is set to first 
          video in file 
        - prior cropping method HLine has been removed
    
    """
    
    #if batch processing, set file to first file to be processed
    video_dict['file'] = video_dict['FileNames'][0] if fstfile else video_dict['file']      
    
    #Upoad file and check that it exists
    video_dict['fpath'] = os.path.join(os.path.normpath(video_dict['dpath']), video_dict['file'])
    if os.path.isfile(video_dict['fpath']):
        print('file: {file}'.format(file=video_dict['fpath']))
        cap = cv2.VideoCapture(video_dict['fpath'])
    else:
        raise FileNotFoundError('{file} not found. Check that directory and file names are correct'.format(
            file=video_dict['fpath']))

    #Print video information. Note that max frame is updated later if fewer frames detected
    cap_max = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
    print('total frames: {frames}'.format(frames=cap_max))
    print('nominal fps: {fps}'.format(fps=cap.get(cv2.CAP_PROP_FPS)))
    print('dimensions (h x w): {h},{w}'.format(
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))))
    
    #check for video p-frames
    if accept_p_frames is False:
        check_p_frames(cap)
    
    #Set first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, video_dict['start']) 
    ret, frame = cap.read() 
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if (video_dict['dsmpl'] < 1):
        frame = cv2.resize(
                    frame,
                    (
                        int(frame.shape[1]*video_dict['dsmpl']),
                        int(frame.shape[0]*video_dict['dsmpl'])
                    ),
                    cv2.INTER_NEAREST)
    video_dict['f0'] = frame
    cap.release()

    #Make first image reference frame on which cropping can be performed
    image = hv.Image((np.arange(frame.shape[1]), np.arange(frame.shape[0]), frame))
    image.opts(
        width=int(frame.shape[1]*video_dict['stretch']['width']),
        height=int(frame.shape[0]*video_dict['stretch']['height']),
        invert_yaxis=True,
        cmap='gray',
        colorbar=True,
        toolbar='below',
        title="First Frame.  Crop if Desired"
    )
    
    #Create polygon element on which to draw and connect via stream to poly drawing tool
    if cropmethod==None:
        image.opts(title="First Frame")
        video_dict['crop'] = None
        return image, video_dict
    
    if cropmethod=='Box':         
        box = hv.Polygons([])
        box.opts(alpha=.5)
        video_dict['crop'] = streams.BoxEdit(source=box,num_objects=1)     
        return (image*box), video_dict
    
    
    
    

########################################################################################

def cropframe(frame, crop=None):
    """ 
    -------------------------------------------------------------------------------------
    
    Crops passed frame with `crop` specification
    
    -------------------------------------------------------------------------------------
    Args:
        frame:: [numpy.ndarray]
            2d numpy array 
        crop:: [hv.streams.stream]
            Holoviews stream object enabling dynamic selection in response to 
            cropping tool. `crop.data` contains x and y coordinates of crop
            boundary vertices. Set to None if no cropping supplied.
    
    -------------------------------------------------------------------------------------
    Returns:
        frame:: [numpy.ndarray]
            2d numpy array
    
    -------------------------------------------------------------------------------------
    Notes:

    """
    
    try:
        Xs=[crop.data['x0'][0],crop.data['x1'][0]]
        Ys=[crop.data['y0'][0],crop.data['y1'][0]]
        fxmin,fxmax=int(min(Xs)), int(max(Xs))
        fymin,fymax=int(min(Ys)), int(max(Ys))
        return frame[fymin:fymax,fxmin:fxmax]
    except:
        return frame
 
    
    
    

########################################################################################

def Reference(video_dict,num_frames=100,
              altfile=False,fstfile=False,frames=None):
    """ 
    -------------------------------------------------------------------------------------
    
    Generates reference frame by taking median of random subset of frames.  This has the 
    effect of removing animal from frame provided animal is not inactive for >=50% of
    the video segment.  
    
    -------------------------------------------------------------------------------------
    Args:
        video_dict:: [dict]
            Dictionary with the following keys:
                'dpath' : directory containing files [str]
                'file' : filename with extension, e.g. 'myvideo.wmv' [str]
                'start' : frame at which to start. 0-based [int]
                'end' : frame at which to end.  set to None if processing 
                        whole video [int]
                'region_names' : list of names of regions.  if no regions, set to None
                'dsmpl' : proptional degree to which video should be downsampled
                        by (0-1).
                'stretch' : Dictionary used to alter display of frames, with the following keys:
                        'width' : proportion by which to stretch frame width [float]
                        'height' : proportion by which to stretch frame height [float]
                        *Does not influence actual processing, unlike dsmpl.
                'reference': Reference image that the current frame is compared to. [numpy.array]
                'roi_stream' : Holoviews stream object enabling dynamic selection in response to 
                               selection tool. `poly_stream.data` contains x and y coordinates of roi 
                               vertices. [hv.streams.stream]
                'crop' : Enables dynamic box selection of cropping parameters.  
                         Holoviews stream object enabling dynamic selection in response to 
                         `stream.data` contains x and y coordinates of crop boundary vertices.
                         [hv.streams.BoxEdit]
                'mask' : [dict]
                    Dictionary with the following keys:
                        'mask' : boolean numpy array identifying regions to exlude
                                 from analysis.  If no such regions, equal to
                                 None. [bool numpy array)   
                        'mask_stream' : Holoviews stream object enabling dynamic selection 
                                in response to selection tool. `mask_stream.data` contains 
                                x and y coordinates of region vertices. [holoviews polystream]
                'scale:: [dict]
                        Dictionary with the following keys:
                            'px_distance' : distance between reference points, in pixels [numeric]
                            'true_distance' : distance between reference points, in desired scale 
                                               (e.g. cm) [numeric]
                            'true_scale' : string containing name of scale (e.g. 'cm') [str]
                            'factor' : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
                'ftype' : (only if batch processing) 
                          video file type extension (e.g. 'wmv') [str]
                'FileNames' : (only if batch processing)
                              List of filenames of videos in folder to be batch 
                              processed.  [list]
                'f0' : (only if batch processing)
                        first frame of video [numpy array]
        
        num_frames:: [uint]
            Number of frames to base reference frame on.
            
        altfile:: [bool]
            Specify whether alternative file than video to be processed will be
            used to generate reference frame. If `altfile=True`, it is expected
            that `video_dict` contains `altfile` key.
        
        fstfile:: [bool]
            Dictates whether to use first file in video_dict['FileNames'] to generate
            reference.  True/False
        
        frames:: [np array]
            User defined selection of frames to use for generating reference
    
    -------------------------------------------------------------------------------------
    Returns:
        reference:: [numpy.array]
            Reference image. Median of random subset of frames.
        image:: [holoviews.image]
            Holoviews Image of reference image.
    
    -------------------------------------------------------------------------------------
    Notes:
        - If `altfile` is specified, it will be used to generate reference.
    
    """
    
    #set file to use for reference
    video_dict['file'] = video_dict['FileNames'][0] if fstfile else video_dict['file']      
    vname = video_dict.get("altfile","") if altfile else video_dict['file']    
    fpath = os.path.join(os.path.normpath(video_dict['dpath']), vname)
    if os.path.isfile(fpath):
        cap = cv2.VideoCapture(fpath)
    else:
        raise FileNotFoundError('File not found. Check that directory and file names are correct.')
    cap.set(cv2.CAP_PROP_POS_FRAMES,0)
    
    #Get video dimensions with any cropping applied
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if (video_dict['dsmpl'] < 1):
        frame = cv2.resize(
                    frame,
                    (
                        int(frame.shape[1]*video_dict['dsmpl']),
                        int(frame.shape[0]*video_dict['dsmpl'])
                    ),
                    cv2.INTER_NEAREST)
    frame = cropframe(
        frame, 
        video_dict.get('crop')
    )
    h,w = frame.shape[0], frame.shape[1]
    cap_max = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
    cap_max = int(video_dict['end']) if video_dict['end'] is not None else cap_max
    
    #Collect subset of frames
    if frames is None:
        #frames = np.random.randint(video_dict['start'],cap_max,num_frames)
        frames = np.linspace(start=video_dict['start'], stop=cap_max, num=num_frames)
    else:
        num_frames = len(frames) #make sure num_frames equals length of passed list
        
    collection = np.zeros((num_frames,h,w))  
    for (idx,framenum) in enumerate(frames):    
        grabbed = False
        while grabbed == False: 
            cap.set(cv2.CAP_PROP_POS_FRAMES, framenum)
            ret, frame = cap.read()
            if ret == True:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if (video_dict['dsmpl'] < 1):
                    gray = cv2.resize(
                        gray,
                        (
                            int(gray.shape[1]*video_dict['dsmpl']),
                            int(gray.shape[0]*video_dict['dsmpl'])
                        ),
                        cv2.INTER_NEAREST)
                gray = cropframe(
                    gray, 
                    video_dict.get('crop')
                )
                collection[idx,:,:]=gray
                grabbed = True
            elif ret == False:
                framenum = np.random.randint(video_dict['start'],cap_max,1)[0]
                pass
    cap.release() 

    reference = np.median(collection,axis=0)
    image = hv.Image((np.arange(reference.shape[1]),
                      np.arange(reference.shape[0]), 
                      reference)).opts(width=int(reference.shape[1]*video_dict['stretch']['width']),
                                       height=int(reference.shape[0]*video_dict['stretch']['height']),
                                       invert_yaxis=True,
                                       cmap='gray',
                                       colorbar=True,
                                       toolbar='below',
                                       title="Reference Frame") 
    return reference, image    





########################################################################################

def Locate(cap,tracking_params,video_dict,prior=None):
    """ 
    -------------------------------------------------------------------------------------
    
    Return location of animal in frame, in x/y coordinates. 
    
    -------------------------------------------------------------------------------------
    Args:
        cap:: [cv2.VideoCapture]
            OpenCV VideoCapture class instance for video.
        
        tracking_params:: [dict]
            Dictionary with the following keys:
                'loc_thresh' : Percentile of difference values below which are set to 0. 
                               After calculating pixel-wise difference between passed 
                               frame and reference frame, these values are tthresholded 
                               to make subsequent defining of center of mass more 
                               reliable. [float between 0-100]
                'use_window' : Will window surrounding prior location be 
                               imposed?  Allows changes in area surrounding animal's 
                               location on previous frame to be more heavily influential
                               in determining animal's current location.
                               After finding pixel-wise difference between passed frame 
                               and reference frame, difference values outside square window 
                               of prior location will be multiplied by (1 - window_weight), 
                               reducing their overall influence. [bool]
                'window_size' : If `use_window=True`, the length of one side of square 
                                window, in pixels. [uint] 
                'window_weight' : 0-1 scale for window, if used, where 1 is maximal 
                                  weight of window surrounding prior locaiton. 
                                  [float between 0-1]
                'method' : 'abs', 'light', or 'dark'.  If 'abs', absolute difference
                           between reference and current frame is taken, and thus the 
                           background of the frame doesn't matter. 'light' specifies that
                           the animal is lighter than the background. 'dark' specifies that 
                           the animal is darker than the background. 
                'rmv_wire' : True/False, indicating whether to use wire removal function.  [bool] 
                'wire_krn' : size of kernel used for morphological opening to remove wire. [int]
                
        video_dict:: [dict]
            Dictionary with the following keys:
                'dpath' : directory containing files [str]
                'file' : filename with extension, e.g. 'myvideo.wmv' [str]
                'start' : frame at which to start. 0-based [int]
                'end' : frame at which to end.  set to None if processing 
                        whole video [int]
                'region_names' : list of names of regions.  if no regions, set to None
                'dsmpl' : proptional degree to which video should be downsampled
                        by (0-1).
                'stretch' : Dictionary used to alter display of frames, with the following keys:
                        'width' : proportion by which to stretch frame width [float]
                        'height' : proportion by which to stretch frame height [float]
                        *Does not influence actual processing, unlike dsmpl.
                'reference': Reference image that the current frame is compared to. [numpy.array]
                'roi_stream' : Holoviews stream object enabling dynamic selection in response to 
                               selection tool. `poly_stream.data` contains x and y coordinates of roi 
                               vertices. [hv.streams.stream]
                'crop' : Enables dynamic box selection of cropping parameters.  
                         Holoviews stream object enabling dynamic selection in response to 
                         `stream.data` contains x and y coordinates of crop boundary vertices.
                         [hv.streams.BoxEdit]
                'mask' : [dict]
                    Dictionary with the following keys:
                        'mask' : boolean numpy array identifying regions to exlude
                                 from analysis.  If no such regions, equal to
                                 None. [bool numpy array)   
                        'mask_stream' : Holoviews stream object enabling dynamic selection 
                                in response to selection tool. `mask_stream.data` contains 
                                x and y coordinates of region vertices. [holoviews polystream]
                'scale:: [dict]
                        Dictionary with the following keys:
                            'px_distance' : distance between reference points, in pixels [numeric]
                            'true_distance' : distance between reference points, in desired scale 
                                               (e.g. cm) [numeric]
                            'true_scale' : string containing name of scale (e.g. 'cm') [str]
                            'factor' : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
                'ftype' : (only if batch processing) 
                          video file type extension (e.g. 'wmv') [str]
                'FileNames' : (only if batch processing)
                              List of filenames of videos in folder to be batch 
                              processed.  [list]
                'f0' : (only if batch processing)
                        first frame of video [numpy array]
        
        prior:: [list]
            If window is being used, list of length 2 is passed, where first index is 
            prior y position, and second index is prior x position.
    
    -------------------------------------------------------------------------------------
    Returns:
        ret:: [bool]
            Specifies whether frame is returned in response to cv2.VideoCapture.read.
        
        dif:: [numpy.array]
            Pixel-wise difference from prior frame, after thresholding and
            applying window weight.
        
        com:: [tuple]
            Indices of center of mass as tuple in the form: (y,x).
        
        frame:: [numpy.array]
            Original video frame after cropping.
    
    -------------------------------------------------------------------------------------
    Notes:
    
    """
    
    #attempt to load frame
    ret, frame = cap.read() 
    
    #set window dimensions
    if prior != None and tracking_params['use_window']==True:
        window_size = tracking_params['window_size']//2
        ymin,ymax = prior[0]-window_size, prior[0]+window_size
        xmin,xmax = prior[1]-window_size, prior[1]+window_size

    if ret == True:
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if (video_dict['dsmpl'] < 1):
            frame = cv2.resize(
                frame,
                (
                    int(frame.shape[1]*video_dict['dsmpl']),
                    int(frame.shape[0]*video_dict['dsmpl'])
                ),
                cv2.INTER_NEAREST)
        frame = cropframe(
            frame,
            video_dict.get('crop')
        )
        
        #find difference from reference
        if tracking_params['method'] == 'abs':
            dif = np.absolute(frame-video_dict['reference'])
        elif tracking_params['method'] == 'light':
            dif = frame-video_dict['reference']
        elif tracking_params['method'] == 'dark':
            dif = video_dict['reference']-frame
        dif = dif.astype('int16')
        if 'mask' in video_dict.keys():
            if video_dict['mask']['mask'] is not None:
                    dif[video_dict['mask']['mask']] = 0
              
        #apply window
        weight = 1 - tracking_params['window_weight']
        if prior != None and tracking_params['use_window']==True:
            dif = dif + (dif.min() * -1) #scale so lowest value is 0
            dif_weights = np.ones(dif.shape)*weight
            dif_weights[slice(ymin if ymin>0 else 0, ymax),
                        slice(xmin if xmin>0 else 0, xmax)]=1
            dif = dif*dif_weights
            
        #threshold differences and find center of mass for remaining values
        dif[dif<np.percentile(dif,tracking_params['loc_thresh'])]=0
        
        #remove influence of wire
        if tracking_params['rmv_wire'] == True:
            ksize = tracking_params['wire_krn']
            kernel = np.ones((ksize,ksize),np.uint8)
            dif_wirermv = cv2.morphologyEx(dif, cv2.MORPH_OPEN, kernel)
            krn_violation =  dif_wirermv.sum()==0
            dif = dif if krn_violation else dif_wirermv
            if krn_violation:
                print("WARNING: wire_krn too large. Reverting to rmv_wire=False for frame {x}".format(
                    x= int(cap.get(cv2.CAP_PROP_POS_FRAMES)-1-video_dict['start'])))
            
        com=ndimage.measurements.center_of_mass(dif)
        return ret, dif, com, frame
    
    else:
        return ret, None, None, frame

    
    
    
    
########################################################################################        

def TrackLocation(video_dict,tracking_params):
    """ 
    -------------------------------------------------------------------------------------
    
    For each frame in video define location of animal, in x/y coordinates, and distance
    travelled from previous frame.
    
    -------------------------------------------------------------------------------------
    Args:
        video_dict:: [dict]
            Dictionary with the following keys:
                'dpath' : directory containing files [str]
                'file' : filename with extension, e.g. 'myvideo.wmv' [str]
                'start' : frame at which to start. 0-based [int]
                'end' : frame at which to end.  set to None if processing 
                        whole video [int]
                'region_names' : list of names of regions.  if no regions, set to None
                'dsmpl' : proptional degree to which video should be downsampled
                        by (0-1).
                'stretch' : Dictionary used to alter display of frames, with the following keys:
                        'width' : proportion by which to stretch frame width [float]
                        'height' : proportion by which to stretch frame height [float]
                        *Does not influence actual processing, unlike dsmpl.
                'reference': Reference image that the current frame is compared to. [numpy.array]
                'roi_stream' : Holoviews stream object enabling dynamic selection in response to 
                               selection tool. `poly_stream.data` contains x and y coordinates of roi 
                               vertices. [hv.streams.stream]
                'crop' : Enables dynamic box selection of cropping parameters.  
                         Holoviews stream object enabling dynamic selection in response to 
                         `stream.data` contains x and y coordinates of crop boundary vertices.
                         [hv.streams.BoxEdit]
                'mask' : [dict]
                    Dictionary with the following keys:
                        'mask' : boolean numpy array identifying regions to exlude
                                 from analysis.  If no such regions, equal to
                                 None. [bool numpy array)   
                        'mask_stream' : Holoviews stream object enabling dynamic selection 
                                in response to selection tool. `mask_stream.data` contains 
                                x and y coordinates of region vertices. [holoviews polystream]
                'scale:: [dict]
                        Dictionary with the following keys:
                            'px_distance' : distance between reference points, in pixels [numeric]
                            'true_distance' : distance between reference points, in desired scale 
                                               (e.g. cm) [numeric]
                            'true_scale' : string containing name of scale (e.g. 'cm') [str]
                            'factor' : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
                'ftype' : (only if batch processing) 
                          video file type extension (e.g. 'wmv') [str]
                'FileNames' : (only if batch processing)
                              List of filenames of videos in folder to be batch 
                              processed.  [list]
                'f0' : (only if batch processing)
                        first frame of video [numpy array] 
                              
        tracking_params:: [dict]
            Dictionary with the following keys:
                'loc_thresh' : Percentile of difference values below which are set to 0. 
                               After calculating pixel-wise difference between passed 
                               frame and reference frame, these values are tthresholded 
                               to make subsequent defining of center of mass more 
                               reliable. [float between 0-100]
                'use_window' : Will window surrounding prior location be 
                               imposed?  Allows changes in area surrounding animal's 
                               location on previous frame to be more heavily influential
                               in determining animal's current location.
                               After finding pixel-wise difference between passed frame 
                               and reference frame, difference values outside square window 
                               of prior location will be multiplied by (1 - window_weight), 
                               reducing their overall influence. [bool]
                'window_size' : If `use_window=True`, the length of one side of square 
                                window, in pixels. [uint] 
                'window_weight' : 0-1 scale for window, if used, where 1 is maximal 
                                  weight of window surrounding prior locaiton. 
                                  [float between 0-1]
                'method' : 'abs', 'light', or 'dark'.  If 'abs', absolute difference
                           between reference and current frame is taken, and thus the 
                           background of the frame doesn't matter. 'light' specifies that
                           the animal is lighter than the background. 'dark' specifies that 
                           the animal is darker than the background. 
                'rmv_wire' : True/False, indicating whether to use wire removal function.  [bool] 
                'wire_krn' : size of kernel used for morphological opening to remove wire. [int]    
    
    -------------------------------------------------------------------------------------
    Returns:
        df:: [pandas.dataframe]
            Pandas dataframe with frame by frame x and y locations,
            distance travelled, as well as video information and parameter values.
    
    -------------------------------------------------------------------------------------
    Notes:
    
    """
          
    #load video
    cap = cv2.VideoCapture(video_dict['fpath'])#set file
    cap.set(cv2.CAP_PROP_POS_FRAMES,video_dict['start']) #set starting frame
    cap_max = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
    cap_max = int(video_dict['end']) if video_dict['end'] is not None else cap_max  
    
    #Initialize vector to store motion values in
    X = np.zeros(cap_max - video_dict['start'])
    Y = np.zeros(cap_max - video_dict['start'])
    D = np.zeros(cap_max - video_dict['start'])

    #Loop through frames to detect frame by frame differences
    time.sleep(.2) #allow printing
    for f in tqdm(range(len(D))):
        
        if f>0: 
            yprior = np.around(Y[f-1]).astype(int)
            xprior = np.around(X[f-1]).astype(int)
            ret,dif,com,frame = Locate(cap,tracking_params,video_dict,prior=[yprior,xprior])
        else:
            ret,dif,com,frame = Locate(cap,tracking_params,video_dict)
                                                
        if ret == True:          
            Y[f] = com[0]
            X[f] = com[1]
            if f>0:
                D[f] = np.sqrt((Y[f]-Y[f-1])**2 + (X[f]-X[f-1])**2)
        else:
            #if no frame is detected
            f = f-1
            X = X[:f] #Amend length of X vector
            Y = Y[:f] #Amend length of Y vector
            D = D[:f] #Amend length of D vector
            break   
            
    #release video
    cap.release()
    time.sleep(.2) #allow printing
    print('total frames processed: {f}\n'.format(f=len(D)))
    
    #create pandas dataframe
    df = pd.DataFrame(
    {'File' : video_dict['file'],
     'Location_Thresh': np.ones(len(D))*tracking_params['loc_thresh'],
     'Use_Window': str(tracking_params['use_window']),
     'Window_Weight': np.ones(len(D))*tracking_params['window_weight'],
     'Window_Size': np.ones(len(D))*tracking_params['window_size'],
     'Start_Frame': np.ones(len(D))*video_dict['start'],
     'Frame': np.arange(len(D)),
     'X': X,
     'Y': Y,
     'Distance_px': D
    })
    
    #add region of interest info
    df = ROI_Location(video_dict, df) 
    if video_dict['region_names'] is not None:
        print('Defining transitions...')
        df['ROI_location'] = ROI_linearize(df[video_dict['region_names']])
        df['ROI_transition'] = ROI_transitions(df['ROI_location'])
    
    #update scale, if known
    df = ScaleDistance(video_dict, df=df, column='Distance_px')
       
    return df





########################################################################################

def LocationThresh_View(video_dict,tracking_params,examples=4):
    """ 
    -------------------------------------------------------------------------------------
    
    Display example tracking with selected parameters for a random subset of frames. 
    NOTE that because individual frames are analyzed independently, weighting 
    based upon prior location is not implemented.
    
    -------------------------------------------------------------------------------------
    Args:
  
        video_dict:: [dict]
            Dictionary with the following keys:
                'dpath' : directory containing files [str]
                'file' : filename with extension, e.g. 'myvideo.wmv' [str]
                'start' : frame at which to start. 0-based [int]
                'end' : frame at which to end.  set to None if processing 
                        whole video [int]
                'region_names' : list of names of regions.  if no regions, set to None
                'dsmpl' : proptional degree to which video should be downsampled
                        by (0-1).
                'stretch' : Dictionary used to alter display of frames, with the following keys:
                        'width' : proportion by which to stretch frame width [float]
                        'height' : proportion by which to stretch frame height [float]
                        *Does not influence actual processing, unlike dsmpl.
                'reference': Reference image that the current frame is compared to. [numpy.array]
                'roi_stream' : Holoviews stream object enabling dynamic selection in response to 
                               selection tool. `poly_stream.data` contains x and y coordinates of roi 
                               vertices. [hv.streams.stream]
                'crop' : Enables dynamic box selection of cropping parameters.  
                         Holoviews stream object enabling dynamic selection in response to 
                         `stream.data` contains x and y coordinates of crop boundary vertices.
                         [hv.streams.BoxEdit]
                'mask' : [dict]
                    Dictionary with the following keys:
                        'mask' : boolean numpy array identifying regions to exlude
                                 from analysis.  If no such regions, equal to
                                 None. [bool numpy array)   
                        'mask_stream' : Holoviews stream object enabling dynamic selection 
                                in response to selection tool. `mask_stream.data` contains 
                                x and y coordinates of region vertices. [holoviews polystream]
                'scale:: [dict]
                        Dictionary with the following keys:
                            'px_distance' : distance between reference points, in pixels [numeric]
                            'true_distance' : distance between reference points, in desired scale 
                                               (e.g. cm) [numeric]
                            'true_scale' : string containing name of scale (e.g. 'cm') [str]
                            'factor' : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
                'ftype' : (only if batch processing) 
                          video file type extension (e.g. 'wmv') [str]
                'FileNames' : (only if batch processing)
                              List of filenames of videos in folder to be batch 
                              processed.  [list]
                'f0' : (only if batch processing)
                        first frame of video [numpy array]
            
        tracking_params:: [dict]
            Dictionary with the following keys:
                'loc_thresh' : Percentile of difference values below which are set to 0. 
                               After calculating pixel-wise difference between passed 
                               frame and reference frame, these values are tthresholded 
                               to make subsequent defining of center of mass more 
                               reliable. [float between 0-100]
                'use_window' : Will window surrounding prior location be 
                               imposed?  Allows changes in area surrounding animal's 
                               location on previous frame to be more heavily influential
                               in determining animal's current location.
                               After finding pixel-wise difference between passed frame 
                               and reference frame, difference values outside square window 
                               of prior location will be multiplied by (1 - window_weight), 
                               reducing their overall influence. [bool]
                'window_size' : If `use_window=True`, the length of one side of square 
                                window, in pixels. [uint] 
                'window_weight' : 0-1 scale for window, if used, where 1 is maximal 
                                  weight of window surrounding prior locaiton. 
                                  [float between 0-1]
                'method' : 'abs', 'light', or 'dark'.  If 'abs', absolute difference
                           between reference and current frame is taken, and thus the 
                           background of the frame doesn't matter. 'light' specifies that
                           the animal is lighter than the background. 'dark' specifies that 
                           the animal is darker than the background. 
                'rmv_wire' : True/False, indicating whether to use wire removal function.  [bool] 
                'wire_krn' : size of kernel used for morphological opening to remove wire. [int] 
                           
        examples:: [uint]
            The number of frames for location tracking to be tested on.
        
    
    -------------------------------------------------------------------------------------
    Returns:
        df:: [holoviews.Layout]
            Returns Holoviews Layout with original images on left and heat plots with 
            animal's estimated position marked on right.
    
    -------------------------------------------------------------------------------------
    Notes:
        - if `stretch` values are modified, this will only influence display and not
          calculation
    
    """
    
    #load video
    cap = cv2.VideoCapture(video_dict['fpath'])
    cap_max = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
    cap_max = int(video_dict['end']) if video_dict['end'] is not None else cap_max
    
    #examine random frames
    images = []
    for example in range (examples):
        
        #analyze frame
        ret = False
        while ret is False:     
            frm=np.random.randint(video_dict['start'],cap_max) #select random frame
            cap.set(cv2.CAP_PROP_POS_FRAMES,frm) #sets frame to be next to be grabbed
            ret,dif,com,frame = Locate(cap, tracking_params, video_dict) 

        #plot original frame
        image_orig = hv.Image((np.arange(frame.shape[1]), np.arange(frame.shape[0]), frame))
        image_orig.opts(
            width=int(video_dict['reference'].shape[1]*video_dict['stretch']['width']),
            height=int(video_dict['reference'].shape[0]*video_dict['stretch']['height']),
            invert_yaxis=True,cmap='gray',toolbar='below',
            title="Frame: " + str(frm))
        orig_overlay = image_orig * hv.Points(([com[1]],[com[0]])).opts(
            color='red',size=20,marker='+',line_width=3) 
        
        #plot heatmap
        dif = dif*(255//dif.max())
        image_heat = hv.Image((
            np.arange(dif.shape[1]), 
            np.arange(dif.shape[0]), 
            dif))
        image_heat.opts(
            width=int(dif.shape[1]*video_dict['stretch']['width']),
            height=int(dif.shape[0]*video_dict['stretch']['height']),
            invert_yaxis=True,cmap='jet',toolbar='below',
            title="Frame: " + str(frm - video_dict['start']))
        heat_overlay = image_heat * hv.Points(([com[1]],[com[0]])).opts(
            color='red',size=20,marker='+',line_width=3) 
        
        images.extend([orig_overlay,heat_overlay])
    
    cap.release()
    layout = hv.Layout(images)
    return layout





########################################################################################    
    
def ROI_plot(video_dict):
    """ 
    -------------------------------------------------------------------------------------
    
    Creates interactive tool for defining regions of interest, based upon array
    `region_names`. If `region_names=None`, reference frame is returned but no regions
    can be drawn.
    
    -------------------------------------------------------------------------------------
    Args:
        video_dict:: [dict]
            Dictionary with the following keys:
                'dpath' : directory containing files [str]
                'file' : filename with extension, e.g. 'myvideo.wmv' [str]
                'start' : frame at which to start. 0-based [int]
                'end' : frame at which to end.  set to None if processing 
                        whole video [int]
                'region_names' : list of names of regions.  if no regions, set to None
                'dsmpl' : proptional degree to which video should be downsampled
                        by (0-1).
                'stretch' : Dictionary used to alter display of frames, with the following keys:
                        'width' : proportion by which to stretch frame width [float]
                        'height' : proportion by which to stretch frame height [float]
                        *Does not influence actual processing, unlike dsmpl.
                'reference': Reference image that the current frame is compared to. [numpy.array]
                'roi_stream' : Holoviews stream object enabling dynamic selection in response to 
                               selection tool. `poly_stream.data` contains x and y coordinates of roi 
                               vertices. [hv.streams.stream]
                'crop' : Enables dynamic box selection of cropping parameters.  
                         Holoviews stream object enabling dynamic selection in response to 
                         `stream.data` contains x and y coordinates of crop boundary vertices.
                         [hv.streams.BoxEdit]
                'mask' : [dict]
                    Dictionary with the following keys:
                        'mask' : boolean numpy array identifying regions to exlude
                                 from analysis.  If no such regions, equal to
                                 None. [bool numpy array)   
                        'mask_stream' : Holoviews stream object enabling dynamic selection 
                                in response to selection tool. `mask_stream.data` contains 
                                x and y coordinates of region vertices. [holoviews polystream]
                'scale:: [dict]
                        Dictionary with the following keys:
                            'px_distance' : distance between reference points, in pixels [numeric]
                            'true_distance' : distance between reference points, in desired scale 
                                               (e.g. cm) [numeric]
                            'true_scale' : string containing name of scale (e.g. 'cm') [str]
                            'factor' : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
                'ftype' : (only if batch processing) 
                          video file type extension (e.g. 'wmv') [str]
                'FileNames' : (only if batch processing)
                              List of filenames of videos in folder to be batch 
                              processed.  [list]
                'f0' : (only if batch processing)
                        first frame of video [numpy array]
                                      
    
    -------------------------------------------------------------------------------------
    Returns:
        image * poly * dmap:: [holoviews.Overlay]
            Reference frame that can be drawn upon to define regions of interest.
        
        poly_stream:: [hv.streams.stream]
            Holoviews stream object enabling dynamic selection in response to 
            selection tool. `poly_stream.data` contains x and y coordinates of roi 
            vertices.
    
    -------------------------------------------------------------------------------------
    Notes:
        - if `stretch` values are modified, this will only influence dispplay and not
          calculation
    
    """
    
    #get number of objects to be drawn
    nobjects = len(video_dict['region_names']) if video_dict['region_names'] else 0 

    #Make reference image the base image on which to draw
    image = hv.Image((
        np.arange(video_dict['reference'].shape[1]),
        np.arange(video_dict['reference'].shape[0]),
        video_dict['reference']))
    image.opts(
        width=int(video_dict['reference'].shape[1]*video_dict['stretch']['width']),
        height=int(video_dict['reference'].shape[0]*video_dict['stretch']['height']),
        invert_yaxis=True,cmap='gray', colorbar=True,toolbar='below',
        title="No Regions to Draw" if nobjects == 0 else "Draw Regions: "+', '.join(video_dict['region_names']))

    #Create polygon element on which to draw and connect via stream to PolyDraw drawing tool
    poly = hv.Polygons([])
    poly_stream = streams.PolyDraw(source=poly, drag=True, num_objects=nobjects, show_vertices=True)
    poly.opts(fill_alpha=0.3, active_tools=['poly_draw'])

    def centers(data):
        try:
            x_ls, y_ls = data['xs'], data['ys']
        except TypeError:
            x_ls, y_ls = [], []
        xs = [np.mean(x) for x in x_ls]
        ys = [np.mean(y) for y in y_ls]
        rois = video_dict['region_names'][:len(xs)]
        return hv.Labels((xs, ys, rois))
    
    if nobjects > 0:
        dmap = hv.DynamicMap(centers, streams=[poly_stream])
        return (image * poly * dmap), poly_stream
    else:
        return (image),None
    

    
    
    
########################################################################################    

def ROI_Location(video_dict, location):
    """ 
    -------------------------------------------------------------------------------------
    
    For each frame, determine which regions of interest the animal is in.  For each
    region of interest, boolean array is added to `location` dataframe passed, with 
    column name being the region name.
    
    -------------------------------------------------------------------------------------
    Args:
        video_dict:: [dict]
            Dictionary with the following keys:
                'dpath' : directory containing files [str]
                'file' : filename with extension, e.g. 'myvideo.wmv' [str]
                'start' : frame at which to start. 0-based [int]
                'end' : frame at which to end.  set to None if processing 
                        whole video [int]
                'region_names' : list of names of regions.  if no regions, set to None
                'dsmpl' : proptional degree to which video should be downsampled
                        by (0-1).
                'stretch' : Dictionary used to alter display of frames, with the following keys:
                        'width' : proportion by which to stretch frame width [float]
                        'height' : proportion by which to stretch frame height [float]
                        *Does not influence actual processing, unlike dsmpl.
                'reference': Reference image that the current frame is compared to. [numpy.array]
                'roi_stream' : Holoviews stream object enabling dynamic selection in response to 
                               selection tool. `poly_stream.data` contains x and y coordinates of roi 
                               vertices. [hv.streams.stream]
                'crop' : Enables dynamic box selection of cropping parameters.  
                         Holoviews stream object enabling dynamic selection in response to 
                         `stream.data` contains x and y coordinates of crop boundary vertices.
                         [hv.streams.BoxEdit]
                'mask' : [dict]
                    Dictionary with the following keys:
                        'mask' : boolean numpy array identifying regions to exlude
                                 from analysis.  If no such regions, equal to
                                 None. [bool numpy array)   
                        'mask_stream' : Holoviews stream object enabling dynamic selection 
                                in response to selection tool. `mask_stream.data` contains 
                                x and y coordinates of region vertices. [holoviews polystream]
                'scale:: [dict]
                        Dictionary with the following keys:
                            'px_distance' : distance between reference points, in pixels [numeric]
                            'true_distance' : distance between reference points, in desired scale 
                                               (e.g. cm) [numeric]
                            'true_scale' : string containing name of scale (e.g. 'cm') [str]
                            'factor' : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
                'ftype' : (only if batch processing) 
                          video file type extension (e.g. 'wmv') [str]
                'FileNames' : (only if batch processing)
                              List of filenames of videos in folder to be batch 
                              processed.  [list]
                'f0' : (only if batch processing)
                        first frame of video [numpy array]
        
        location:: [pandas.dataframe]
            Pandas dataframe with frame by frame x and y locations,
            distance travelled, as well as video information and parameter values. 
            Must contain column names 'X' and 'Y'.

    -------------------------------------------------------------------------------------
    Returns:
        location:: [pandas.dataframe]
            For each region of interest, boolean array is added to `location` dataframe 
            passed, with column name being the region name. Additionally, under column
            `ROI_coordinates`, coordinates of vertices of each region of interest are
            printed. This takes the form of a dictionary of x and y coordinates, e.g.:
                'xs' : [[region 1 x coords], [region 2 x coords]],
                'ys' : [[region 1 y coords], [region 2 y coords]]
                                      
    -------------------------------------------------------------------------------------
    Notes:
    
    """
    
    if video_dict['region_names'] == None:
        return location

    #Create ROI Masks
    ROI_masks = {}
    for poly in range(len(video_dict['roi_stream'].data['xs'])):
        x = np.array(video_dict['roi_stream'].data['xs'][poly]) #x coordinates
        y = np.array(video_dict['roi_stream'].data['ys'][poly]) #y coordinates
        xy = np.column_stack((x,y)).astype('uint64') #xy coordinate pairs
        mask = np.zeros(video_dict['reference'].shape) # create empty mask
        cv2.fillPoly(mask, pts =[xy], color=255) #fill polygon  
        ROI_masks[video_dict['region_names'][poly]] = mask==255 #save to ROI masks as boolean 

    #Create arrays to store whether animal is within given ROI
    ROI_location = {}
    for mask in ROI_masks:
        ROI_location[mask]=np.full(len(location['Frame']),False,dtype=bool)

    #For each frame assess truth of animal being in each ROI
    for f in location['Frame']:
        y,x = location['Y'][f], location['X'][f]
        for mask in ROI_masks:
            ROI_location[mask][f] = ROI_masks[mask][int(y),int(x)]
    
    #Add data to location data frame
    for x in ROI_location:
        location[x]=ROI_location[x]
    
    #Add ROI coordinates
    location['ROI_coordinates']=str(video_dict['roi_stream'].data)
    
    return location





########################################################################################        

def ROI_linearize(rois, null_name = 'non_roi'):
    
    """ 
    -------------------------------------------------------------------------------------
    
    Creates array defining ROI as string for each frame
    
    -------------------------------------------------------------------------------------
    Args:
        rois:: [pd.DataFrame]
            Pandas dataframe where each column corresponds to an ROI, with boolean values
            defining if animal is in said roi.
        null_name:: [string]
            Name used when animals is not in any defined roi.
    
    -------------------------------------------------------------------------------------
    Returns:
        rois['ROI_location']:: [pd.Series]
            pd.Series defining ROI as string for each frame
    
    -------------------------------------------------------------------------------------
    Notes:
    
    """
    region_names = rois.columns.values
    rois['ROI_location'] = null_name
    for region in region_names:
        rois['ROI_location'][rois[region]] = rois['ROI_location'][rois[region]].apply(
            lambda x: '_'.join([x, region]) if x!=null_name else region
        )
    return rois['ROI_location']






########################################################################################        

def ROI_transitions(regions, include_first=False):
    """ 
    -------------------------------------------------------------------------------------
    
    Creates boolean array defining where transitions between each ROI occur.
    
    -------------------------------------------------------------------------------------
    Args:
        regions:: [Pandas Series]
            Pandas Series defining ROI as string for each frame
        include_first:: [string]
            Whether to count first frame as transition
    
    -------------------------------------------------------------------------------------
    Returns:
        transitions:: [Boolean array]
            pd.Series defining where transitions between ROIs occur.
    
    -------------------------------------------------------------------------------------
    Notes:
    
    """
    regions_offset = np.append(regions[0], regions[0:-1])
    transitions = regions!=regions_offset
    if include_first:
        transitions[0] = True
    return transitions





########################################################################################        
    
def Summarize_Location(location, video_dict, bin_dict=None):
    """ 
    -------------------------------------------------------------------------------------
    
    Generates summary of distance travelled and proportional time spent in each region
    of interest according to user defined time bins.  If bins are not provided 
    (`bin_dict=None`), average of entire video segment will be provided.
    
    -------------------------------------------------------------------------------------
    Args:
        location:: [pandas.dataframe]
            Pandas dataframe with frame by frame x and y locations,
            distance travelled, as well as video information and parameter values. 
            Additionally, for each region of interest, boolean array indicating whether 
            animal is in the given region for each frame.
      
        video_dict:: [dict]
            Dictionary with the following keys:
                'dpath' : directory containing files [str]
                'file' : filename with extension, e.g. 'myvideo.wmv' [str]
                'start' : frame at which to start. 0-based [int]
                'end' : frame at which to end.  set to None if processing 
                        whole video [int]
                'region_names' : list of names of regions.  if no regions, set to None
                'dsmpl' : proptional degree to which video should be downsampled
                        by (0-1).
                'stretch' : Dictionary used to alter display of frames, with the following keys:
                        'width' : proportion by which to stretch frame width [float]
                        'height' : proportion by which to stretch frame height [float]
                        *Does not influence actual processing, unlike dsmpl.
                'reference': Reference image that the current frame is compared to. [numpy.array]
                'roi_stream' : Holoviews stream object enabling dynamic selection in response to 
                               selection tool. `poly_stream.data` contains x and y coordinates of roi 
                               vertices. [hv.streams.stream]
                'crop' : Enables dynamic box selection of cropping parameters.  
                         Holoviews stream object enabling dynamic selection in response to 
                         `stream.data` contains x and y coordinates of crop boundary vertices.
                         [hv.streams.BoxEdit]
                'mask' : [dict]
                    Dictionary with the following keys:
                        'mask' : boolean numpy array identifying regions to exlude
                                 from analysis.  If no such regions, equal to
                                 None. [bool numpy array)   
                        'mask_stream' : Holoviews stream object enabling dynamic selection 
                                in response to selection tool. `mask_stream.data` contains 
                                x and y coordinates of region vertices. [holoviews polystream]
                'scale:: [dict]
                        Dictionary with the following keys:
                            'px_distance' : distance between reference points, in pixels [numeric]
                            'true_distance' : distance between reference points, in desired scale 
                                               (e.g. cm) [numeric]
                            'true_scale' : string containing name of scale (e.g. 'cm') [str]
                            'factor' : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
                'ftype' : (only if batch processing) 
                          video file type extension (e.g. 'wmv') [str]
                'FileNames' : (only if batch processing)
                              List of filenames of videos in folder to be batch 
                              processed.  [list]
                'f0' : (only if batch processing)
                        first frame of video [numpy array]
                              
        bin_dict:: [dict]
            Dictionary specifying bins.  Dictionary keys should be names of the bins.  
            Dictionary value for each bin should be a tuple, with the start and end of 
            the bin, in seconds, relative to the start of the analysis period 
            (i.e. if start frame is 100, it will be relative to that). If no bins are to 
            be specified, set bin_dict = None.
            example: bin_dict = {1:(0,100), 2:(100,200)}                             

    
    -------------------------------------------------------------------------------------
    Returns:
        bins:: [pandas.dataframe]
            Pandas dataframe with distance travelled and proportional time spent in each 
            region of interest according to user defined time bins, as well as video 
            information and parameter values. If no region names are supplied 
            (`region_names=None`), only distance travelled will be included.
                                      
    -------------------------------------------------------------------------------------
    Notes:
    
    """
    
    #define bins
    avg_dict = {'all': (location['Frame'].min(), location['Frame'].max())}   
    bin_dict = bin_dict if bin_dict is not None else avg_dict
    
    #get summary info
    bins = (pd.Series(bin_dict).rename('range(f)')
            .reset_index().rename(columns=dict(index='bin')))    
    bins['Distance_px'] = bins['range(f)'].apply(
        lambda r: location[location['Frame'].between(*r)]['Distance_px'].sum())
    if video_dict['region_names'] is not None:
        bins_reg = bins['range(f)'].apply(
            lambda r: location[location['Frame'].between(*r)][video_dict['region_names']].mean())
        bins = bins.join(bins_reg)
        drp_cols = ['Distance_px', 'Frame', 'X', 'Y'] + video_dict['region_names']
    else:
        drp_cols = ['Distance_px', 'Frame', 'X', 'Y']
    bins = pd.merge(
        location.drop(drp_cols, axis='columns'),
        bins,
        left_index=True,
        right_index=True)
    
    #scale distance
    bins = ScaleDistance(video_dict,df=bins,column='Distance_px') 
    
    return bins





######################################################################################## 

def Batch_LoadFiles(video_dict):
    """ 
    -------------------------------------------------------------------------------------
    
    Populates list of files in directory (`dpath`) that are of the specified file type
    (`ftype`).  List is held in `video_dict['FileNames']`.
    
    -------------------------------------------------------------------------------------
    Args:
        video_dict:: [dict]
            Dictionary with the following keys:
                'dpath' : directory containing files [str]
                'file' : filename with extension, e.g. 'myvideo.wmv' [str]
                'start' : frame at which to start. 0-based [int]
                'end' : frame at which to end.  set to None if processing 
                        whole video [int]
                'region_names' : list of names of regions.  if no regions, set to None
                'dsmpl' : proptional degree to which video should be downsampled
                        by (0-1).
                'stretch' : Dictionary used to alter display of frames, with the following keys:
                        'width' : proportion by which to stretch frame width [float]
                        'height' : proportion by which to stretch frame height [float]
                        *Does not influence actual processing, unlike dsmpl.
                'reference': Reference image that the current frame is compared to. [numpy.array]
                'roi_stream' : Holoviews stream object enabling dynamic selection in response to 
                               selection tool. `poly_stream.data` contains x and y coordinates of roi 
                               vertices. [hv.streams.stream]
                'crop' : Enables dynamic box selection of cropping parameters.  
                         Holoviews stream object enabling dynamic selection in response to 
                         `stream.data` contains x and y coordinates of crop boundary vertices.
                         [hv.streams.BoxEdit]
                'mask' : [dict]
                    Dictionary with the following keys:
                        'mask' : boolean numpy array identifying regions to exlude
                                 from analysis.  If no such regions, equal to
                                 None. [bool numpy array)   
                        'mask_stream' : Holoviews stream object enabling dynamic selection 
                                in response to selection tool. `mask_stream.data` contains 
                                x and y coordinates of region vertices. [holoviews polystream]
                'scale:: [dict]
                        Dictionary with the following keys:
                            'px_distance' : distance between reference points, in pixels [numeric]
                            'true_distance' : distance between reference points, in desired scale 
                                               (e.g. cm) [numeric]
                            'true_scale' : string containing name of scale (e.g. 'cm') [str]
                            'factor' : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
                'ftype' : (only if batch processing) 
                          video file type extension (e.g. 'wmv') [str]
                'FileNames' : (only if batch processing)
                              List of filenames of videos in folder to be batch 
                              processed.  [list]
                'f0' : (only if batch processing)
                        first frame of video [numpy array]

    
    -------------------------------------------------------------------------------------
    Returns:
        video_dict:: [dict]
            Dictionary with the following keys:
                'dpath' : directory containing files [str]
                'file' : filename with extension, e.g. 'myvideo.wmv' [str]
                'start' : frame at which to start. 0-based [int]
                'end' : frame at which to end.  set to None if processing 
                        whole video [int]
                'region_names' : list of names of regions.  if no regions, set to None
                'dsmpl' : proptional degree to which video should be downsampled
                        by (0-1).
                'stretch' : Dictionary used to alter display of frames, with the following keys:
                        'width' : proportion by which to stretch frame width [float]
                        'height' : proportion by which to stretch frame height [float]
                        *Does not influence actual processing, unlike dsmpl.
                'reference': Reference image that the current frame is compared to. [numpy.array]
                'roi_stream' : Holoviews stream object enabling dynamic selection in response to 
                               selection tool. `poly_stream.data` contains x and y coordinates of roi 
                               vertices. [hv.streams.stream]
                'crop' : Enables dynamic box selection of cropping parameters.  
                         Holoviews stream object enabling dynamic selection in response to 
                         `stream.data` contains x and y coordinates of crop boundary vertices.
                         [hv.streams.BoxEdit]
                'mask' : [dict]
                    Dictionary with the following keys:
                        'mask' : boolean numpy array identifying regions to exlude
                                 from analysis.  If no such regions, equal to
                                 None. [bool numpy array)   
                        'mask_stream' : Holoviews stream object enabling dynamic selection 
                                in response to selection tool. `mask_stream.data` contains 
                                x and y coordinates of region vertices. [holoviews polystream]
                'scale:: [dict]
                        Dictionary with the following keys:
                            'px_distance' : distance between reference points, in pixels [numeric]
                            'true_distance' : distance between reference points, in desired scale 
                                               (e.g. cm) [numeric]
                            'true_scale' : string containing name of scale (e.g. 'cm') [str]
                            'factor' : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
                'ftype' : (only if batch processing) 
                          video file type extension (e.g. 'wmv') [str]
                'FileNames' : (only if batch processing)
                              List of filenames of videos in folder to be batch 
                              processed.  [list]
                'f0' : (only if batch processing)
                        first frame of video [numpy array]
    
    -------------------------------------------------------------------------------------
    Notes:
    
    """

    #Get list of video files of designated type
    if os.path.isdir(video_dict['dpath']):
        video_dict['FileNames'] = sorted(os.listdir(video_dict['dpath']))
        video_dict['FileNames'] = fnmatch.filter(video_dict['FileNames'], ('*.' + video_dict['ftype'])) 
        return video_dict
    else:
        raise FileNotFoundError('{path} not found. Check that directory is correct'.format(
            path=video_dict['dpath']))

        
        
        
        
######################################################################################## 

def Batch_Process(video_dict,tracking_params,bin_dict,accept_p_frames=False):   
    """ 
    -------------------------------------------------------------------------------------
    
    Run LocationTracking on folder of videos of specified filetype. 
    
    -------------------------------------------------------------------------------------
    Args:
        video_dict:: [dict]
            Dictionary with the following keys:
                'dpath' : directory containing files [str]
                'file' : filename with extension, e.g. 'myvideo.wmv' [str]
                'start' : frame at which to start. 0-based [int]
                'end' : frame at which to end.  set to None if processing 
                        whole video [int]
                'region_names' : list of names of regions.  if no regions, set to None
                'dsmpl' : proptional degree to which video should be downsampled
                        by (0-1).
                'stretch' : Dictionary used to alter display of frames, with the following keys:
                        'width' : proportion by which to stretch frame width [float]
                        'height' : proportion by which to stretch frame height [float]
                        *Does not influence actual processing, unlike dsmpl.
                'reference': Reference image that the current frame is compared to. [numpy.array]
                'roi_stream' : Holoviews stream object enabling dynamic selection in response to 
                               selection tool. `poly_stream.data` contains x and y coordinates of roi 
                               vertices. [hv.streams.stream]
                'crop' : Enables dynamic box selection of cropping parameters.  
                         Holoviews stream object enabling dynamic selection in response to 
                         `stream.data` contains x and y coordinates of crop boundary vertices.
                         [hv.streams.BoxEdit]
                'mask' : [dict]
                    Dictionary with the following keys:
                        'mask' : boolean numpy array identifying regions to exlude
                                 from analysis.  If no such regions, equal to
                                 None. [bool numpy array)   
                        'mask_stream' : Holoviews stream object enabling dynamic selection 
                                in response to selection tool. `mask_stream.data` contains 
                                x and y coordinates of region vertices. [holoviews polystream]
                'scale:: [dict]
                        Dictionary with the following keys:
                            'px_distance' : distance between reference points, in pixels [numeric]
                            'true_distance' : distance between reference points, in desired scale 
                                               (e.g. cm) [numeric]
                            'true_scale' : string containing name of scale (e.g. 'cm') [str]
                            'factor' : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
                'ftype' : (only if batch processing) 
                          video file type extension (e.g. 'wmv') [str]
                'FileNames' : (only if batch processing)
                              List of filenames of videos in folder to be batch 
                              processed.  [list]
                'f0' : (only if batch processing)
                        first frame of video [numpy array]
        
        tracking_params:: [dict]
            Dictionary with the following keys:
                'loc_thresh' : Percentile of difference values below which are set to 0. 
                               After calculating pixel-wise difference between passed 
                               frame and reference frame, these values are tthresholded 
                               to make subsequent defining of center of mass more 
                               reliable. [float between 0-100]
                'use_window' : Will window surrounding prior location be 
                               imposed?  Allows changes in area surrounding animal's 
                               location on previous frame to be more heavily influential
                               in determining animal's current location.
                               After finding pixel-wise difference between passed frame 
                               and reference frame, difference values outside square window 
                               of prior location will be multiplied by (1 - window_weight), 
                               reducing their overall influence. [bool]
                'window_size' : If `use_window=True`, the length of one side of square 
                                window, in pixels. [uint] 
                'window_weight' : 0-1 scale for window, if used, where 1 is maximal 
                                  weight of window surrounding prior locaiton. 
                                  [float between 0-1]
                'method' : 'abs', 'light', or 'dark'.  If 'abs', absolute difference
                           between reference and current frame is taken, and thus the 
                           background of the frame doesn't matter. 'light' specifies that
                           the animal is lighter than the background. 'dark' specifies that 
                           the animal is darker than the background. 
                'rmv_wire' : True/False, indicating whether to use wire removal function.  [bool] 
                'wire_krn' : size of kernel used for morphological opening to remove wire. [int]
                
         accept_p_frames::[bool]
            Dictates whether to allow videos with temporal compresssion.  Currenntly, if
            more than 1/100 frames returns false, error is flagged.
    
    -------------------------------------------------------------------------------------
    Returns:
        summary_all:: [pandas.dataframe]
            Pandas dataframe with distance travelled and proportional time spent in each 
            region of interest according to user defined time bins, as well as video 
            information and parameter values. If no region names are supplied 
            (`region_names=None`), only distance travelled will be included.
            
        layout:: [hv.Layout]
            Holoviews layout wherein for each session the reference frame is returned
            with the regions of interest highlightted and the animals location across
            the session overlaid atop the reference image.
    
    -------------------------------------------------------------------------------------
    Notes:
    
    """
    
    images = []
    for file in video_dict['FileNames']:
        
        print ('Processing File: {f}'.format(f=file))  
        video_dict['file'] = file 
        video_dict['fpath'] = os.path.join(os.path.normpath(video_dict['dpath']), file)
        
        #Print video information. Note that max frame is updated later if fewer frames detected
        cap = cv2.VideoCapture(video_dict['fpath'])
        cap_max = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
        print('total frames: {frames}'.format(frames=cap_max))
        print('nominal fps: {fps}'.format(fps=cap.get(cv2.CAP_PROP_FPS)))
        print('dimensions (h x w): {h},{w}'.format(
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))))
        
        #check for video p-frames
        if accept_p_frames is False:
            check_p_frames(cap)
        
        video_dict['reference'], image = Reference(video_dict,num_frames=50) 
        location = TrackLocation(video_dict,tracking_params)
        location.to_csv(os.path.splitext(video_dict['fpath'])[0] + '_LocationOutput.csv', index=False)
        file_summary = Summarize_Location(location, video_dict, bin_dict=bin_dict)
               
        try: 
            summary_all = pd.concat([summary_all,file_summary],sort=False)
        except NameError: 
            summary_all = file_summary
        
        trace = showtrace(video_dict,location)
        heatmap = Heatmap(video_dict, location, sigma=None)
        images = images + [(trace.opts(title=file)), (heatmap.opts(title=file))]

    #Write summary data to csv file
    sum_pathout = os.path.join(os.path.normpath(video_dict['dpath']), 'BatchSummary.csv')
    summary_all.to_csv(sum_pathout, index=False)
    
    layout = hv.Layout(images)
    return summary_all, layout





########################################################################################        

def PlayVideo(video_dict,display_dict,location):  
    """ 
    -------------------------------------------------------------------------------------
    
    Play portion of video back, displaying animal's estimated location. Video is played
    in notebook

    -------------------------------------------------------------------------------------
    Args:
        video_dict:: [dict]
            Dictionary with the following keys:
                'dpath' : directory containing files [str]
                'file' : filename with extension, e.g. 'myvideo.wmv' [str]
                'start' : frame at which to start. 0-based [int]
                'end' : frame at which to end.  set to None if processing 
                        whole video [int]
                'region_names' : list of names of regions.  if no regions, set to None
                'dsmpl' : proptional degree to which video should be downsampled
                        by (0-1).
                'stretch' : Dictionary used to alter display of frames, with the following keys:
                        'width' : proportion by which to stretch frame width [float]
                        'height' : proportion by which to stretch frame height [float]
                        *Does not influence actual processing, unlike dsmpl.
                'reference': Reference image that the current frame is compared to. [numpy.array]
                'roi_stream' : Holoviews stream object enabling dynamic selection in response to 
                               selection tool. `poly_stream.data` contains x and y coordinates of roi 
                               vertices. [hv.streams.stream]
                'crop' : Enables dynamic box selection of cropping parameters.  
                         Holoviews stream object enabling dynamic selection in response to 
                         `stream.data` contains x and y coordinates of crop boundary vertices.
                         [hv.streams.BoxEdit]
                'mask' : [dict]
                    Dictionary with the following keys:
                        'mask' : boolean numpy array identifying regions to exlude
                                 from analysis.  If no such regions, equal to
                                 None. [bool numpy array)   
                        'mask_stream' : Holoviews stream object enabling dynamic selection 
                                in response to selection tool. `mask_stream.data` contains 
                                x and y coordinates of region vertices. [holoviews polystream]
                'scale:: [dict]
                        Dictionary with the following keys:
                            'px_distance' : distance between reference points, in pixels [numeric]
                            'true_distance' : distance between reference points, in desired scale 
                                               (e.g. cm) [numeric]
                            'true_scale' : string containing name of scale (e.g. 'cm') [str]
                            'factor' : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
                'ftype' : (only if batch processing) 
                          video file type extension (e.g. 'wmv') [str]
                'FileNames' : (only if batch processing)
                              List of filenames of videos in folder to be batch 
                              processed.  [list]
                'f0' : (only if batch processing)
                        first frame of video [numpy array]
                
        display_dict:: [dict]
            Dictionary with the following keys:
                'start' : start point of video segment in frames [int]
                'end' : end point of video segment in frames [int]
                'resize' : Default is None, in which original size is retained.
                           Alternatively, set to tuple as follows: (width,height).
                           Because this is in pixel units, must be integer values.
                'fps' : frames per second of video file/files to be processed [int]
                'save_video' : option to save video if desired [bool]
                               Currently, will be saved at 20 fps even if video 
                               is something else
                               
        location:: [pandas.dataframe]
            Pandas dataframe with frame by frame x and y locations,
            distance travelled, as well as video information and parameter values. 
            Additionally, for each region of interest, boolean array indicating whether 
            animal is in the given region for each frame. 
            
    
    -------------------------------------------------------------------------------------
    Returns:
        Nothing returned
    
    -------------------------------------------------------------------------------------
    Notes:

    """


    #Load Video and Set Saving Parameters
    cap = cv2.VideoCapture(video_dict['fpath'])#set file\
    if display_dict['save_video']==True:
        ret, frame = cap.read() #read frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if (video_dict['dsmpl'] < 1):
            frame = cv2.resize(
                frame,
                (
                    int(frame.shape[1]*video_dict['dsmpl']),
                    int(frame.shape[0]*video_dict['dsmpl'])
                ),
                cv2.INTER_NEAREST)
        frame = cropframe(frame, video_dict['crop'])
        height, width = int(frame.shape[0]), int(frame.shape[1])
        fourcc = 0#cv2.VideoWriter_fourcc(*'jpeg') #only writes up to 20 fps, though video read can be 30.
        writer = cv2.VideoWriter(os.path.join(os.path.normpath(video_dict['dpath']), 'video_output.avi'), 
                                 fourcc, 20.0, 
                                 (width, height),
                                 isColor=False)

    #Initialize video play options   
    cap.set(cv2.CAP_PROP_POS_FRAMES,video_dict['start']+display_dict['start']) 

    #Play Video
    for f in range(display_dict['start'],display_dict['stop']):
        ret, frame = cap.read() #read frame
        if ret == True:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if (video_dict['dsmpl'] < 1):
                frame = cv2.resize(
                    frame,
                    (
                        int(frame.shape[1]*video_dict['dsmpl']),
                        int(frame.shape[0]*video_dict['dsmpl'])
                    ),
                    cv2.INTER_NEAREST)
            frame = cropframe(frame, video_dict['crop'])
            markposition = (int(location['X'][f]),int(location['Y'][f]))
            cv2.drawMarker(img=frame,position=markposition,color=255)
            display_image(frame,display_dict['fps'],display_dict['resize'])
            #Save video (if desired). 
            if display_dict['save_video']==True:
                writer.write(frame) 
        if ret == False:
            print('warning. failed to get video frame')

    #Close video window and video writer if open
    print('Done playing segment')
    if display_dict['save_video']==True:
        writer.release()

def display_image(frame,fps,resize):
    img = PIL.Image.fromarray(frame, "L")
    img = img.resize(size=resize) if resize else img
    buffer = BytesIO()
    img.save(buffer,format="JPEG")    
    display(Image(data=buffer.getvalue()))
    time.sleep(1/fps)
    clear_output(wait=True)

    
    

    
########################################################################################

def PlayVideo_ext(video_dict,display_dict,location,crop=None):  
    """ 
    -------------------------------------------------------------------------------------
    
    Play portion of video back, displaying animal's estimated location

    -------------------------------------------------------------------------------------
    Args:
        video_dict:: [dict]
            Dictionary with the following keys:
                'dpath' : directory containing files [str]
                'file' : filename with extension, e.g. 'myvideo.wmv' [str]
                'start' : frame at which to start. 0-based [int]
                'end' : frame at which to end.  set to None if processing 
                        whole video [int]
                'region_names' : list of names of regions.  if no regions, set to None
                'dsmpl' : proptional degree to which video should be downsampled
                        by (0-1).
                'stretch' : Dictionary used to alter display of frames, with the following keys:
                        'width' : proportion by which to stretch frame width [float]
                        'height' : proportion by which to stretch frame height [float]
                        *Does not influence actual processing, unlike dsmpl.
                'reference': Reference image that the current frame is compared to. [numpy.array]
                'roi_stream' : Holoviews stream object enabling dynamic selection in response to 
                               selection tool. `poly_stream.data` contains x and y coordinates of roi 
                               vertices. [hv.streams.stream]
                'crop' : Enables dynamic box selection of cropping parameters.  
                         Holoviews stream object enabling dynamic selection in response to 
                         `stream.data` contains x and y coordinates of crop boundary vertices.
                         [hv.streams.BoxEdit]
                'mask' : [dict]
                    Dictionary with the following keys:
                        'mask' : boolean numpy array identifying regions to exlude
                                 from analysis.  If no such regions, equal to
                                 None. [bool numpy array)   
                        'mask_stream' : Holoviews stream object enabling dynamic selection 
                                in response to selection tool. `mask_stream.data` contains 
                                x and y coordinates of region vertices. [holoviews polystream]
                'scale:: [dict]
                        Dictionary with the following keys:
                            'px_distance' : distance between reference points, in pixels [numeric]
                            'true_distance' : distance between reference points, in desired scale 
                                               (e.g. cm) [numeric]
                            'true_scale' : string containing name of scale (e.g. 'cm') [str]
                            'factor' : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
                'ftype' : (only if batch processing) 
                          video file type extension (e.g. 'wmv') [str]
                'FileNames' : (only if batch processing)
                              List of filenames of videos in folder to be batch 
                              processed.  [list]
                'f0' : (only if batch processing)
                        first frame of video [numpy array]
                
        display_dict:: [dict]
            Dictionary with the following keys:
                'start' : start point of video segment in frames [int]
                'end' : end point of video segment in frames [int]
                'fps' : frames per second of video file/files to be processed [int]
                'save_video' : option to save video if desired [bool]
                               Currently, will be saved at 20 fps even if video 
                               is something else
                               
        location:: [pandas.dataframe]
            Pandas dataframe with frame by frame x and y locations,
            distance travelled, as well as video information and parameter values. 
            Additionally, for each region of interest, boolean array indicating whether 
            animal is in the given region for each frame. 
  
    
    -------------------------------------------------------------------------------------
    Returns:
        Nothing returned
    
    -------------------------------------------------------------------------------------
    Notes:

    """

    #Load Video and Set Saving Parameters
    cap = cv2.VideoCapture(video_dict['fpath'])#set file\
    if display_dict['save_video']==True:
        ret, frame = cap.read() #read frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if (video_dict['dsmpl'] < 1):
            frame = cv2.resize(
                frame,
                (
                    int(frame.shape[1]*video_dict['dsmpl']),
                    int(frame.shape[0]*video_dict['dsmpl'])
                ),
                cv2.INTER_NEAREST)
        frame = cropframe(frame, crop)
        height, width = int(frame.shape[0]), int(frame.shape[1])
        fourcc = 0#cv2.VideoWriter_fourcc(*'jpeg') #only writes up to 20 fps, though video read can be 30.
        writer = cv2.VideoWriter(os.path.join(os.path.normpath(video_dict['dpath']), 'video_output.avi'), 
                                 fourcc, 20.0, 
                                 (width, height),
                                 isColor=False)

    #Initialize video play options   
    cap.set(cv2.CAP_PROP_POS_FRAMES,video_dict['start']+display_dict['start']) 
    rate = int(1000/display_dict['fps']) 

    #Play Video
    for f in range(display_dict['start'],display_dict['stop']):
        ret, frame = cap.read() #read frame
        if ret == True:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if (video_dict['dsmpl'] < 1):
                frame = cv2.resize(
                    frame,
                    (
                        int(frame.shape[1]*video_dict['dsmpl']),
                        int(frame.shape[0]*video_dict['dsmpl'])
                    ),
                    cv2.INTER_NEAREST)
            frame = cropframe(frame, crop)
            markposition = (int(location['X'][f]),int(location['Y'][f]))
            cv2.drawMarker(img=frame,position=markposition,color=255)
            cv2.imshow("preview",frame)
            cv2.waitKey(rate)
            #Save video (if desired). 
            if display_dict['save_video']==True:
                writer.write(frame) 
        if ret == False:
            print('warning. failed to get video frame')

    #Close video window and video writer if open        
    cv2.destroyAllWindows()
    _=cv2.waitKey(1) 
    if display_dict['save_video']==True:
        writer.release()

    
    
    
    
########################################################################################

def showtrace(video_dict, location, color="red",alpha=.8,size=3):
    """ 
    -------------------------------------------------------------------------------------
    
    Create image where animal location across session is displayed atop reference frame

    -------------------------------------------------------------------------------------
    Args:
        
        video_dict:: [dict]
            Dictionary with the following keys:
                'dpath' : directory containing files [str]
                'file' : filename with extension, e.g. 'myvideo.wmv' [str]
                'start' : frame at which to start. 0-based [int]
                'end' : frame at which to end.  set to None if processing 
                        whole video [int]
                'region_names' : list of names of regions.  if no regions, set to None
                'dsmpl' : proptional degree to which video should be downsampled
                        by (0-1).
                'stretch' : Dictionary used to alter display of frames, with the following keys:
                        'width' : proportion by which to stretch frame width [float]
                        'height' : proportion by which to stretch frame height [float]
                        *Does not influence actual processing, unlike dsmpl.
                'reference': Reference image that the current frame is compared to. [numpy.array]
                'roi_stream' : Holoviews stream object enabling dynamic selection in response to 
                               selection tool. `poly_stream.data` contains x and y coordinates of roi 
                               vertices. [hv.streams.stream]
                'crop' : Enables dynamic box selection of cropping parameters.  
                         Holoviews stream object enabling dynamic selection in response to 
                         `stream.data` contains x and y coordinates of crop boundary vertices.
                         [hv.streams.BoxEdit]
                'mask' : [dict]
                    Dictionary with the following keys:
                        'mask' : boolean numpy array identifying regions to exlude
                                 from analysis.  If no such regions, equal to
                                 None. [bool numpy array)   
                        'mask_stream' : Holoviews stream object enabling dynamic selection 
                                in response to selection tool. `mask_stream.data` contains 
                                x and y coordinates of region vertices. [holoviews polystream]
                'scale:: [dict]
                        Dictionary with the following keys:
                            'px_distance' : distance between reference points, in pixels [numeric]
                            'true_distance' : distance between reference points, in desired scale 
                                               (e.g. cm) [numeric]
                            'true_scale' : string containing name of scale (e.g. 'cm') [str]
                            'factor' : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
                'ftype' : (only if batch processing) 
                          video file type extension (e.g. 'wmv') [str]
                'FileNames' : (only if batch processing)
                              List of filenames of videos in folder to be batch 
                              processed.  [list]
                'f0' : (only if batch processing)
                        first frame of video [numpy array]
        
        location:: [pandas.dataframe]
            Pandas dataframe with frame by frame x and y locations,
            distance travelled, as well as video information and parameter values. 

            
        color:: [str]
            Color of trace.  See Holoviews documentation for color options
                               
        alpha:: [float]
            Alpha of trace.  See Holoviews documentation for details
        
        size:: [float]
            Size of trace.  See Holoviews documentation for details.     
    
    -------------------------------------------------------------------------------------
    Returns:
        holoviews.Overlay
            Location of animal superimposed upon reference. If poly_stream is passed
            than regions of interest will also be outlined.
    
    -------------------------------------------------------------------------------------
    Notes:

    """
    
    video_dict['roi_stream'] = video_dict['roi_stream'] if 'roi_stream' in video_dict else None
    if video_dict['roi_stream'] != None:
        lst = []
        for poly in range(len(video_dict['roi_stream'].data['xs'])):
            x = np.array(video_dict['roi_stream'].data['xs'][poly]) #x coordinates
            y = np.array(video_dict['roi_stream'].data['ys'][poly]) #y coordinates
            lst.append( [ (x[vert],y[vert]) for vert in range(len(x)) ] )
        poly = hv.Polygons(lst).opts(fill_alpha=0.1,line_dash='dashed')
        
    image = hv.Image((np.arange(video_dict['reference'].shape[1]),
                      np.arange(video_dict['reference'].shape[0]),
                      video_dict['reference'])
                    ).opts(width=int(video_dict['reference'].shape[1]*video_dict['stretch']['width']),
                           height=int(video_dict['reference'].shape[0]*video_dict['stretch']['height']),
                           invert_yaxis=True,cmap='gray',toolbar='below',
                           title="Motion Trace")
    
    points = hv.Scatter(np.array([location['X'],location['Y']]).T).opts(color='red',alpha=alpha,size=size)
    
    return (image*poly*points) if video_dict['roi_stream']!=None else (image*points)





########################################################################################    

def Heatmap (video_dict, location, sigma=None):
    """ 
    -------------------------------------------------------------------------------------
    
    Create heatmap of relative time in each location. Max value is set to maxiumum
    in any one location.

    -------------------------------------------------------------------------------------
    Args:
        
        video_dict:: [dict]
            Dictionary with the following keys:
                'dpath' : directory containing files [str]
                'file' : filename with extension, e.g. 'myvideo.wmv' [str]
                'start' : frame at which to start. 0-based [int]
                'end' : frame at which to end.  set to None if processing 
                        whole video [int]
                'region_names' : list of names of regions.  if no regions, set to None
                'dsmpl' : proptional degree to which video should be downsampled
                        by (0-1).
                'stretch' : Dictionary used to alter display of frames, with the following keys:
                        'width' : proportion by which to stretch frame width [float]
                        'height' : proportion by which to stretch frame height [float]
                        *Does not influence actual processing, unlike dsmpl.
                'reference': Reference image that the current frame is compared to. [numpy.array]
                'roi_stream' : Holoviews stream object enabling dynamic selection in response to 
                               selection tool. `poly_stream.data` contains x and y coordinates of roi 
                               vertices. [hv.streams.stream]
                'crop' : Enables dynamic box selection of cropping parameters.  
                         Holoviews stream object enabling dynamic selection in response to 
                         `stream.data` contains x and y coordinates of crop boundary vertices.
                         [hv.streams.BoxEdit]
                'mask' : [dict]
                    Dictionary with the following keys:
                        'mask' : boolean numpy array identifying regions to exlude
                                 from analysis.  If no such regions, equal to
                                 None. [bool numpy array)   
                        'mask_stream' : Holoviews stream object enabling dynamic selection 
                                in response to selection tool. `mask_stream.data` contains 
                                x and y coordinates of region vertices. [holoviews polystream]
                'scale:: [dict]
                        Dictionary with the following keys:
                            'px_distance' : distance between reference points, in pixels [numeric]
                            'true_distance' : distance between reference points, in desired scale 
                                               (e.g. cm) [numeric]
                            'true_scale' : string containing name of scale (e.g. 'cm') [str]
                            'factor' : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
                'ftype' : (only if batch processing) 
                          video file type extension (e.g. 'wmv') [str]
                'FileNames' : (only if batch processing)
                              List of filenames of videos in folder to be batch 
                              processed.  [list]
                'f0' : (only if batch processing)
                        first frame of video [numpy array]
        
        location:: [pandas.dataframe]
            Pandas dataframe with frame by frame x and y locations,
            distance travelled, as well as video information and parameter values. 
                
        sigma:: [numeric]
            Optional number specifying sigma of guassian filter
  
    
    -------------------------------------------------------------------------------------
    Returns:
        map_i:: [holoviews.Image]
            Heatmap image
    
    -------------------------------------------------------------------------------------
    Notes:
        stretch only affects display

    """    
    heatmap = np.zeros(video_dict['reference'].shape)
    for frame in range(len(location)):
        Y,X = int(location.Y[frame]), int(location.X[frame])
        heatmap[Y,X]+=1
    
    sigma = np.mean(heatmap.shape)*.05 if sigma == None else sigma
    heatmap = cv2.GaussianBlur(heatmap,(0,0),sigma)
    heatmap = (heatmap / heatmap.max())*255
    
    map_i = hv.Image((np.arange(heatmap.shape[1]), np.arange(heatmap.shape[0]), heatmap))
    map_i.opts(width=int(heatmap.shape[1]*video_dict['stretch']['width']),
           height=int(heatmap.shape[0]*video_dict['stretch']['height']),
           invert_yaxis=True, cmap='jet', alpha=1,
           colorbar=False, toolbar='below', title="Heatmap")
    
    return map_i





########################################################################################    

def DistanceTool(video_dict):
    """ 
    -------------------------------------------------------------------------------------
    
    Creates interactive tool for measuring length between two points, in pixel units, in 
    order to ease process of converting pixel distance measurements to some other scale.
    Use point drawing tool to calculate distance beteen any two popints.
    
    -------------------------------------------------------------------------------------
    Args:
        
        video_dict:: [dict]
            Dictionary with the following keys:
                'dpath' : directory containing files [str]
                'file' : filename with extension, e.g. 'myvideo.wmv' [str]
                'start' : frame at which to start. 0-based [int]
                'end' : frame at which to end.  set to None if processing 
                        whole video [int]
                'region_names' : list of names of regions.  if no regions, set to None
                'dsmpl' : proptional degree to which video should be downsampled
                        by (0-1).
                'stretch' : Dictionary used to alter display of frames, with the following keys:
                        'width' : proportion by which to stretch frame width [float]
                        'height' : proportion by which to stretch frame height [float]
                        *Does not influence actual processing, unlike dsmpl.
                'reference': Reference image that the current frame is compared to. [numpy.array]
                'roi_stream' : Holoviews stream object enabling dynamic selection in response to 
                               selection tool. `poly_stream.data` contains x and y coordinates of roi 
                               vertices. [hv.streams.stream]
                'crop' : Enables dynamic box selection of cropping parameters.  
                         Holoviews stream object enabling dynamic selection in response to 
                         `stream.data` contains x and y coordinates of crop boundary vertices.
                         [hv.streams.BoxEdit]
                'mask' : [dict]
                    Dictionary with the following keys:
                        'mask' : boolean numpy array identifying regions to exlude
                                 from analysis.  If no such regions, equal to
                                 None. [bool numpy array)   
                        'mask_stream' : Holoviews stream object enabling dynamic selection 
                                in response to selection tool. `mask_stream.data` contains 
                                x and y coordinates of region vertices. [holoviews polystream]
                'scale:: [dict]
                        Dictionary with the following keys:
                            'px_distance' : distance between reference points, in pixels [numeric]
                            'true_distance' : distance between reference points, in desired scale 
                                               (e.g. cm) [numeric]
                            'true_scale' : string containing name of scale (e.g. 'cm') [str]
                            'factor' : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
                'ftype' : (only if batch processing) 
                          video file type extension (e.g. 'wmv') [str]
                'FileNames' : (only if batch processing)
                              List of filenames of videos in folder to be batch 
                              processed.  [list]
                'f0' : (only if batch processing)
                        first frame of video [numpy array]
        
    
    -------------------------------------------------------------------------------------
    Returns:
        image * points * dmap:: [holoviews.Overlay]
            Reference frame that can be drawn upon to define 2 points, the distance 
            between which will be measured and displayed.
        
        distance:: [dict]
            Dictionary with the following keys:
                'd' : Euclidean distance between two reference points, in pixel units, 
                      rounded to thousandth. Returns None if no less than 2 points have 
                      been selected.
    
    -------------------------------------------------------------------------------------
    Notes:
        - if `stretch` values are modified, this will only influence dispplay and not
          calculation
    
    """

    #Make reference image the base image on which to draw
    image = hv.Image((
        np.arange(video_dict['reference'].shape[1]), 
        np.arange(video_dict['reference'].shape[0]), 
        video_dict['reference']))
    image.opts(width=int(video_dict['reference'].shape[1]*video_dict['stretch']['width']),
               height=int(video_dict['reference'].shape[0]*video_dict['stretch']['height']),
              invert_yaxis=True,cmap='gray',
              colorbar=True,
               toolbar='below',
              title="Select Points")

    #Create Point instance on which to draw and connect via stream to pointDraw drawing tool 
    points = hv.Points([]).opts(active_tools=['point_draw'], color='red',size=10)
    pointDraw_stream = streams.PointDraw(source=points,num_objects=2) 
    
    def markers(data, distance):
        try:
            x_ls, y_ls = data['x'], data['y']
        except TypeError:
            x_ls, y_ls = [], []
        
        x_ctr, y_ctr = np.mean(x_ls), np.mean(y_ls)
        if len(x_ls) > 1:
            x_dist = (x_ls[0] - x_ls[1])
            y_dist = (y_ls[0] - y_ls[1])
            distance['px_distance'] = np.around( (x_dist**2 + y_dist**2)**(1/2), 3)
            text = "{dist} px".format(dist=distance['px_distance'])
        return hv.Labels((x_ctr, y_ctr, text if len(x_ls) > 1 else "")).opts(
            text_color='blue',text_font_size='14pt')
    
    distance = dict(px_distance=None)
    markers_ptl = fct.partial(markers, distance=distance)
    dmap = hv.DynamicMap(markers_ptl, streams=[pointDraw_stream])
    return (image * points * dmap), distance


########################################################################################

def setScale(distance, scale, scale_dict):

    """ 
    -------------------------------------------------------------------------------------
    
    Updates dictionary with scale information, given the true distance between points 
    (e.g. 100), and the scale unit (e.g. 'cm')
    
    -------------------------------------------------------------------------------------
    Args:
    
        distance :: [numeric]
            The real-world distance between the selected points
        
        scale :: [string]
            The scale used for defining the real world distance.  Can be any string
            (e.g. 'cm', 'in', 'inch', 'stone')

        scale_dict :: [dict]
            Dictionary with the following keys:
                'px_distance' : distance between reference points, in pixels [numeric]
                'true_distance' : distance between reference points, in desired scale 
                                   (e.g. cm) [numeric]
                'true_scale' : string containing name of scale (e.g. 'cm') [str]
                'factor' : ratio of desired scale to pixel (e.g. cm/pixel [numeric]

    -------------------------------------------------------------------------------------
    Returns:
        scale_dict :: [dict]
                Dictionary with the following keys:
                    'px_distance' : distance between reference points, in pixels [numeric]
                    'true_distance' : distance between reference points, in desired scale 
                                       (e.g. cm) [numeric]
                    'true_scale' : string containing name of scale (e.g. 'cm') [str]
                    'factor' : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
    -------------------------------------------------------------------------------------
    Notes:

    """

    scale_dict['true_distance'] = distance
    scale_dict['true_scale'] = scale
    return scale_dict
    


########################################################################################    

def ScaleDistance(video_dict, df=None, column=None):
    """ 
    -------------------------------------------------------------------------------------
    
    Adds column to dataframe by multiplying existing column by scaling factor to change
    scale. Used in order to convert distance from pixel scale to desired real world 
    distance scale.
    
    -------------------------------------------------------------------------------------
    Args:

        video_dict:: [dict]
            Dictionary with the following keys:
                'dpath' : directory containing files [str]
                'file' : filename with extension, e.g. 'myvideo.wmv' [str]
                'start' : frame at which to start. 0-based [int]
                'end' : frame at which to end.  set to None if processing 
                        whole video [int]
                'region_names' : list of names of regions.  if no regions, set to None
                'dsmpl' : proptional degree to which video should be downsampled
                        by (0-1).
                'stretch' : Dictionary used to alter display of frames, with the following keys:
                        'width' : proportion by which to stretch frame width [float]
                        'height' : proportion by which to stretch frame height [float]
                        *Does not influence actual processing, unlike dsmpl.
                'reference': Reference image that the current frame is compared to. [numpy.array]
                'roi_stream' : Holoviews stream object enabling dynamic selection in response to 
                               selection tool. `poly_stream.data` contains x and y coordinates of roi 
                               vertices. [hv.streams.stream]
                'crop' : Enables dynamic box selection of cropping parameters.  
                         Holoviews stream object enabling dynamic selection in response to 
                         `stream.data` contains x and y coordinates of crop boundary vertices.
                         [hv.streams.BoxEdit]
                'mask' : [dict]
                    Dictionary with the following keys:
                        'mask' : boolean numpy array identifying regions to exlude
                                 from analysis.  If no such regions, equal to
                                 None. [bool numpy array)   
                        'mask_stream' : Holoviews stream object enabling dynamic selection 
                                in response to selection tool. `mask_stream.data` contains 
                                x and y coordinates of region vertices. [holoviews polystream]
                'scale:: [dict]
                        Dictionary with the following keys:
                            'px_distance' : distance between reference points, in pixels [numeric]
                            'true_distance' : distance between reference points, in desired scale 
                                               (e.g. cm) [numeric]
                            'true_scale' : string containing name of scale (e.g. 'cm') [str]
                            'factor' : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
                'ftype' : (only if batch processing) 
                          video file type extension (e.g. 'wmv') [str]
                'FileNames' : (only if batch processing)
                              List of filenames of videos in folder to be batch 
                              processed.  [list]
                'f0' : (only if batch processing)
                        first frame of video [numpy array]
        
        df:: [pandas.dataframe]
            Pandas dataframe with column to be scaled.
        
        column:: [str]
            Name of column in df to be scaled
        
    -------------------------------------------------------------------------------------
    Returns:
        df:: [pandas.dataframe]
            Pandas dataframe with column of scaled distance values.
    
    -------------------------------------------------------------------------------------
    Notes:
        - if `stretch` values are modified, this will only influence dispplay and not
          calculation
    
    """
    
    if 'scale' not in video_dict.keys():
        return df

    if video_dict['scale']['px_distance']!= None:
        video_dict['scale']['factor'] = video_dict['scale']['true_distance']/video_dict['scale']['px_distance']
        new_column = "_".join(['Distance', video_dict['scale']['true_scale']])
        df[new_column] = df[column]*video_dict['scale']['factor']
        order = [col for col in df if col not in [column,new_column]]
        order = order + [column,new_column]
        df = df[order]
    else:
        print('Distance between reference points undefined. Cannot scale column: {c}.\
        Returning original dataframe'.format(c=column))
    return df



########################################################################################    
    
def Mask_select(video_dict, fstfile=False):
    """ 
    -------------------------------------------------------------------------------------
    
    Creates interactive tool for defining regions of interest, based upon array
    `region_names`. If `region_names=None`, reference frame is returned but no regions
    can be drawn.
    
    -------------------------------------------------------------------------------------
    Args:
        video_dict:: [dict]
            Dictionary with the following keys:
                'dpath' : directory containing files [str]
                'file' : filename with extension, e.g. 'myvideo.wmv' [str]
                'start' : frame at which to start. 0-based [int]
                'end' : frame at which to end.  set to None if processing 
                        whole video [int]
                'region_names' : list of names of regions.  if no regions, set to None
                'dsmpl' : proptional degree to which video should be downsampled
                        by (0-1).
                'stretch' : Dictionary used to alter display of frames, with the following keys:
                        'width' : proportion by which to stretch frame width [float]
                        'height' : proportion by which to stretch frame height [float]
                        *Does not influence actual processing, unlike dsmpl.
                'reference': Reference image that the current frame is compared to. [numpy.array]
                'roi_stream' : Holoviews stream object enabling dynamic selection in response to 
                               selection tool. `poly_stream.data` contains x and y coordinates of roi 
                               vertices. [hv.streams.stream]
                'crop' : Enables dynamic box selection of cropping parameters.  
                         Holoviews stream object enabling dynamic selection in response to 
                         `stream.data` contains x and y coordinates of crop boundary vertices.
                         [hv.streams.BoxEdit]
                'mask' : [dict]
                    Dictionary with the following keys:
                        'mask' : boolean numpy array identifying regions to exlude
                                 from analysis.  If no such regions, equal to
                                 None. [bool numpy array)   
                        'mask_stream' : Holoviews stream object enabling dynamic selection 
                                in response to selection tool. `mask_stream.data` contains 
                                x and y coordinates of region vertices. [holoviews polystream]
                'scale:: [dict]
                        Dictionary with the following keys:
                            'px_distance' : distance between reference points, in pixels [numeric]
                            'true_distance' : distance between reference points, in desired scale 
                                               (e.g. cm) [numeric]
                            'true_scale' : string containing name of scale (e.g. 'cm') [str]
                            'factor' : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
                'ftype' : (only if batch processing) 
                          video file type extension (e.g. 'wmv') [str]
                'FileNames' : (only if batch processing)
                              List of filenames of videos in folder to be batch 
                              processed.  [list]
                'f0' : (only if batch processing)
                        first frame of video [numpy array]

        fstfile:: [bool]
            Dictates whether to use first file in video_dict['FileNames'] to generate
            reference.  True/False
    
    -------------------------------------------------------------------------------------
    Returns:
        image * poly * dmap:: [holoviews.Overlay]
            First frame of video that can be drawn upon to define regions of interest.
            
        mask:: [dict]
            Dictionary with the following keys:
                'mask' : boolean numpy array identifying regions to exlude
                         from analysis.  If no such regions, equal to
                         None. [bool numpy array)   
                'mask_stream' : Holoviews stream object enabling dynamic selection 
                        in response to selection tool. `mask_stream.data` contains 
                        x and y coordinates of region vertices. [holoviews polystream]
    
    -------------------------------------------------------------------------------------
    Notes:
        - if `stretch` values are modified, this will only influence dispplay and not
          calculation
    
    """
    
    #Load first file if batch processing
    if fstfile:
        video_dict['file'] = video_dict['FileNames'][0] 
        video_dict['fpath'] = os.path.join(os.path.normpath(video_dict['dpath']), video_dict['file'])
        if os.path.isfile(video_dict['fpath']):
            print('file: {file}'.format(file=video_dict['fpath']))
            cap = cv2.VideoCapture(video_dict['fpath'])
        else:
            raise FileNotFoundError('{file} not found. Check that directory and file names are correct'.format(
                file=video_dict['fpath']))
        cap.set(cv2.CAP_PROP_POS_FRAMES, video_dict['start']) 
        ret, frame = cap.read() 
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if (video_dict['dsmpl'] < 1):
            frame = cv2.resize(
                frame,
                (
                    int(frame.shape[1]*video_dict['dsmpl']),
                    int(frame.shape[0]*video_dict['dsmpl'])
                ),
                cv2.INTER_NEAREST)
        video_dict['f0'] = frame
    
    #Make first image the base image on which to draw
    f0 = cropframe(
        video_dict['f0'],
        video_dict.get('crop')
    )
    image = hv.Image((np.arange(f0.shape[1]), np.arange(f0.shape[0]), f0))
    image.opts(width=int(f0.shape[1]*video_dict['stretch']['width']),
               height=int(f0.shape[0]*video_dict['stretch']['height']),
              invert_yaxis=True,cmap='gray',
              colorbar=True,
               toolbar='below',
              title="Draw Regions to be Exluded")

    #Create polygon element on which to draw and connect via stream to PolyDraw drawing tool
    mask = dict(mask=None)
    poly = hv.Polygons([])
    mask['stream'] = streams.PolyDraw(source=poly, drag=True, show_vertices=True)
    #poly_stream = streams.PolyDraw(source=poly, drag=True, show_vertices=True)
    poly.opts(fill_alpha=0.3, active_tools=['poly_draw'])
    points = hv.Points([]).opts(active_tools=['point_draw'], color='red',size=10)
    pointDraw_stream = streams.PointDraw(source=points,num_objects=2) 
    
    def make_mask(data, mask):
        try:
            x_ls, y_ls = data['xs'], data['ys'] 
        except TypeError:
            x_ls, y_ls = [], []
        
        if len(x_ls)>0:
            mask['mask'] = np.zeros(f0.shape) 
            for submask in range(len(x_ls)):
                x = np.array(mask['stream'].data['xs'][submask]) #x coordinates
                y = np.array(mask['stream'].data['ys'][submask]) #y coordinates
                xy = np.column_stack((x,y)).astype('uint64') #xy coordinate pairs
                cv2.fillPoly(mask['mask'], pts =[xy], color=1) #fill polygon  
            mask['mask'] = mask['mask'].astype('bool')
        return hv.Labels((0,0,""))
    
    
    make_mask_ptl = fct.partial(make_mask, mask=mask)        
    dmap = hv.DynamicMap(make_mask_ptl, streams=[mask['stream']])
    return image*poly*dmap, mask



def check_p_frames(cap, p_prop_allowed=.01, frames_checked=300):
    """ 
    -------------------------------------------------------------------------------------
    
    Checks whether video contains substantial portion of p/blank frames
    
    -------------------------------------------------------------------------------------
    Args:
        cap:: [cv2.videocapture]
            OpenCV video capture object.
        p_prop_allowed:: [numeric]
            Proportion of putative p-frames permitted.  Alternatively, proportion of 
            frames permitted to return False when grabbed.
        frames_checked:: [numeric]
            Number of frames to scan for p/blank frames.  If video is shorter
            than number of frames specified, will use number of frames in video.
    
    -------------------------------------------------------------------------------------
    Returns:
    
    -------------------------------------------------------------------------------------
    Notes:
    
    """
    
    frames_checked = min(frames_checked, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    p_allowed = int(frames_checked*p_prop_allowed)
    
    p_frms = 0
    for i in range(frames_checked):
        ret, frame = cap.read()
        p_frms = p_frms+1 if ret==False else p_frms
    if p_frms>p_allowed:
        raise RuntimeError(
            'Video compression method not supported. ' + \
            'Approximately {p}% frames are p frames or blank. '.format(
                p=(p_frms/frames_checked)*100) + \
            'Consider video conversion.')


########################################################################################        
#Code to export svg
#conda install -c conda-forge selenium phantomjs

#import os
#from bokeh import models
#from bokeh.io import export_svgs

#bokeh_obj = hv.renderer('bokeh').get_plot(image).state
#bokeh_obj.output_backend = 'svg'
#export_svgs(bokeh_obj, dpath + '/' + 'Calibration_Frame.svg')

    