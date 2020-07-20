"""

LIST OF FUNCTIONS

LoadAndCrop  
Measure_Motion 
cropframe
Measure_Freezing 
Play_Video 
Play_Video_ext
Save_Data 
Summarize 
Batch 
Calibrate 

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
from scipy import ndimage
import holoviews as hv
from holoviews import opts
from holoviews import streams
from holoviews.streams import Stream, param
from io import BytesIO
from IPython.display import clear_output, Image, display
hv.notebook_extension('bokeh')
warnings.filterwarnings("ignore")





########################################################################################        

def LoadAndCrop(video_dict,stretch={'width':1,'height':1},cropmethod=None,fstfile=False):
    """ 
    -------------------------------------------------------------------------------------
    
    Loads video and creates interactive cropping tool from first frame. In the 
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
                'fps' : frames per second of video files to be processed [int]
                'start' : frame at which to start. 0-based [int]
                'end' : frame at which to end.  set to None if processing 
                        whole video [int]
                'dsmpl' : proptional degree to which video should be downsampled
                        by (0-1).
                'ftype' : (only if batch processing) 
                          video file type extension (e.g. 'wmv') [str]
                'FileNames' : (only if batch processing)
                              List of filenames of videos in folder to be batch 
                              processed.  [list]
                
        stretch:: [dict]
            Dictionary with the following keys:
                'width' : proportion by which to stretch frame width [float]
                'height' : proportion by which to stretch frame height [float]
                
        cropmethod:: [str]
            Method of cropping video.  cropmethod takes the following values:
                None : No cropping 
                'Box' : Create box selection tool for cropping video
                
        fstfile:: [bool]
            Dictates whether to use first file in video_dict['FileNames'] to generate
            reference.  True/False
    
    -------------------------------------------------------------------------------------
    Returns:
        image:: [holoviews.Image]
            Holoviews hv.Image displaying first frame
            
        stream:: [holoviews.streams.stream]
            Holoviews stream object enabling dynamic selection in response to 
            cropping tool. `stream.data` contains x and y coordinates of crop
            boundary vertices.
            
        video_dict:: [dict]
            Dictionary with the following keys:
                'dpath' : directory containing files [str]
                'file' : filename with extension, e.g. 'myvideo.wmv' [str]
                'fps' : frames per second of video file/files to be processed [int]
                'start' : frame at which to start. 0-based [int]
                'end' : frame at which to end.  set to None if processing whole 
                        video [int]
                'dsmpl' : proptional degree to which video should be downsampled
                        by (0-1).
                'ftype' : (only if batch processing) 
                          video file type extension (e.g. 'wmv') [str]
                'FileNames' : (only if batch processing)
                              List of filenames of videos in folder to be 
                              batch processed.  [list]
    
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

    #Get maxiumum frame of file. Note that max frame is updated later if fewer frames detected
    cap_max = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
    print('total frames: {frames}'.format(frames=cap_max))

    #Set first frame. 
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES,video_dict['start']) 
    except:
        cap.set(cv2.CAP_PROP_POS_FRAMES,0)
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
    cap.release() 

    #Make first image reference frame on which cropping can be performed
    image = hv.Image((np.arange(frame.shape[1]), np.arange(frame.shape[0]), frame))
    image.opts(width=int(frame.shape[1]*stretch['width']),
               height=int(frame.shape[0]*stretch['height']),
              invert_yaxis=True,cmap='gray',
              colorbar=True,
               toolbar='below',
              title="First Frame.  Crop if Desired")
    
    #Create polygon element on which to draw and connect via stream to poly drawing tool
    if cropmethod==None:
        image.opts(title="First Frame")
        return image,None,video_dict
    
    if cropmethod=='Box':         
        box = hv.Polygons([])
        box.opts(alpha=.5)
        box_stream = streams.BoxEdit(source=box,num_objects=1)     
        return (image*box),box_stream,video_dict  
    

    
    
    
########################################################################################

def Measure_Motion (video_dict,mt_cutoff,crop=None,SIGMA=1):
    """ 
    -------------------------------------------------------------------------------------
    
    Loops through segment of video file, frame by frame, and calculates number of pixels 
    per frame whose intensity value changed from prior frame.
    
    -------------------------------------------------------------------------------------
    Args:
        video_dict:: [dict]
            Dictionary with the following keys:
                'dpath' : directory containing files [str]
                'file' : filename with extension, e.g. 'myvideo.wmv' [str]
                'fps' : frames per second of video files to be processed [int]
                'start' : frame at which to start. 0-based [int]
                'end' : frame at which to end.  set to None if processing 
                        whole video [int]
                'dsmpl' : proptional degree to which video should be downsampled
                        by (0-1).
                'ftype' : (only if batch processing) 
                          video file type extension (e.g. 'wmv') [str]
                'FileNames' : (only if batch processing)
                              List of filenames of videos in folder to be batch 
                              processed.  [list]
                
        mt_cutoff:: [float]
            Threshold value for determining magnitude of change sufficient to mark
            pixel as changing from prior frame.
                
        crop:: [holoviews.streams.stream]
            Holoviews stream object enabling dynamic selection in response to 
            cropping tool. `crop.data` contains x and y coordinates of crop
            boundary vertices.
                
        SIGMA:: [float]
            Sigma value for gaussian filter applied to each image. Passed to 
            OpenCV `cv2.GuassianBlur`.
    
    -------------------------------------------------------------------------------------
    Returns:
        Motion:: [numpy.array]
            Array containing number of pixels per frame whose intensity change from
            previous frame exceeds `mt_cutoff`. Length is number of frames passed to
            function to loop through. Value of first index, corresponding to first frame,
            is set to 0.
    
    -------------------------------------------------------------------------------------
    Notes:

    """
    
    #Upoad file
    cap = cv2.VideoCapture(video_dict['fpath'])
    cap_max = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
    cap_max = int(video_dict['end']) if video_dict['end'] is not None else cap_max
    cap.set(cv2.CAP_PROP_POS_FRAMES,video_dict['start']) 

    #Initialize first frame and array to store motion values in
    ret, frame_new = cap.read()
    frame_new = cv2.cvtColor(frame_new, cv2.COLOR_BGR2GRAY)
    if (video_dict['dsmpl'] < 1):
        frame_new = cv2.resize(
                    frame_new,
                    (
                        int(frame_new.shape[1]*video_dict['dsmpl']),
                        int(frame_new.shape[0]*video_dict['dsmpl'])
                    ),
                    cv2.INTER_NEAREST)
    frame_new = cropframe(frame_new, crop)
    frame_new = cv2.GaussianBlur(frame_new.astype('float'),(0,0),SIGMA)  
    Motion = np.zeros(cap_max - video_dict['start'])

    #Loop through frames to detect frame by frame differences
    for x in range (1,len(Motion)):
        frame_old = frame_new
        ret, frame_new = cap.read()
        if ret == True:
            #Reset new frame and process calculate difference between old/new frames
            frame_new = cv2.cvtColor(frame_new, cv2.COLOR_BGR2GRAY)
            if (video_dict['dsmpl'] < 1):
                frame_new = cv2.resize(
                    frame_new,
                    (
                        int(frame_new.shape[1]*video_dict['dsmpl']),
                        int(frame_new.shape[0]*video_dict['dsmpl'])
                    ),
                    cv2.INTER_NEAREST)
            frame_new = cropframe(frame_new, crop)
            frame_new = cv2.GaussianBlur(frame_new.astype('float'),(0,0),SIGMA)  
            frame_dif = np.absolute(frame_new - frame_old)
            frame_cut = (frame_dif > mt_cutoff).astype('uint8')
            Motion[x]=np.sum(frame_cut)
        else: 
            #if no frame is detected
            x = x-1 #Reset x to last frame detected
            Motion = Motion[:x] #Amend length of motion vector
            break
        
    cap.release() #release video
    return(Motion) #return motion values





########################################################################################

def cropframe(frame,crop=None):
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

def Measure_Freezing(Motion,FreezeThresh,MinDuration=0):
    """ 
    -------------------------------------------------------------------------------------
    
    Calculates freezing on a frame by frame basis based upon measure of motion.

    -------------------------------------------------------------------------------------
    Args:
        Motion:: [numpy.array]
            Array containing number of pixels per frame whose intensity change from
            previous frame exceeds `mt_cutoff`. 
                
        FreezeThresh:: [float]
            Threshold value for determining magnitude of activity in `Motion` to designate
            frame as freezing/not freezing (i.e. if motion is below `FreezeThresh`, animal
            is likely freezing).
                
        MinDuration:: [uint8]
            Duration for which `Motion` must be below `FreezeThresh` for freezing to be 
            registered.
    
    -------------------------------------------------------------------------------------
    Returns:
        Freezing:: [numpy.array]
            Array defining whether animal is freezing on frame by frame basis.  
            0 = Not Freezing; 100 = Freezing
    
    -------------------------------------------------------------------------------------
    Notes:
        - Although Motion argument is often `Motion` array returned by function 
          `Measure_Motion`, any unidimensional array could be passed.

    """

    #Find frames below thresh
    BelowThresh = (Motion<FreezeThresh).astype(int)

    #Perform local cumulative thresh detection
    #For each consecutive frame motion is below threshold count is increased by 1 until motion goes above thresh, 
    #at which point coint is set back to 0
    CumThresh = np.zeros(len(Motion))
    for x in range (1,len(Motion)):
        if (BelowThresh[x]==1):
            CumThresh[x] = CumThresh[x-1] + BelowThresh[x]

    #Define periods where motion is below thresh for minduration as freezing
    Freezing = (CumThresh>=MinDuration).astype(int)
    for x in range( len(Freezing) - 2, -1, -1) : 
        if Freezing[x] == 0 and Freezing[x+1]>0 and Freezing[x+1]<MinDuration:
            Freezing[x] = Freezing[x+1] + 1
    Freezing = (Freezing>0).astype(int)
    Freezing = Freezing*100 #Convert to Percentage
    
    return(Freezing)





########################################################################################

def PlayVideo(video_dict,display_dict,Freezing,mt_cutoff,crop=None,SIGMA=1):
    """ 
    -------------------------------------------------------------------------------------
    
    Play portion of video back, displaying thresholded video in tandem. Displayed
    externally

    -------------------------------------------------------------------------------------
    Args:
        video_dict:: [dict]
            Dictionary with the following keys:
                'dpath' : directory containing files [str]
                'file' : filename with extension, e.g. 'myvideo.wmv' [str]
                'fps' : frames per second of video files to be processed [int]
                'start' : frame at which to start. 0-based [int]
                'end' : frame at which to end.  set to None if processing 
                        whole video [int]
                'dsmpl' : proptional degree to which video should be downsampled
                        by (0-1).
                'ftype' : (only if batch processing) 
                          video file type extension (e.g. 'wmv') [str]
                'FileNames' : (only if batch processing)
                              List of filenames of videos in folder to be batch 
                              processed.  [list]
                
        display_dict:: [dict]
            Dictionary with the following keys:
                'start' : start point of video segment in frames [int]
                'end' : end point of video segment in frames [int]
                'fps' : frames per second of video file/files to be processed [int]
                'save_video' : option to save video if desired [bool]
                               Currently, will be saved at 20 fps even if video 
                               is something else
                               
        Freezing:: [numpy.array]
            Array defining whether animal is freezing on frame by frame basis.  
            0 = Not Freezing; 100 = Freezing
        
        mt_cutoff:: [float]
            Threshold value for determining magnitude of change sufficient to mark
            pixel as changing from prior frame.
        
        crop:: [holoviews.streams.stream]
            Holoviews stream object enabling dynamic selection in response to 
            cropping tool. `crop.data` contains x and y coordinates of crop
            boundary vertices. Set to None if no cropping supplied.
            
        SIGMA:: [float]
            Sigma value for gaussian filter applied to each image. Passed to 
            OpenCV `cv2.GuassianBlur`.
    
    -------------------------------------------------------------------------------------
    Returns:
        Nothing returned
    
    -------------------------------------------------------------------------------------
    Notes:

    """
    
    #Upoad file
    cap = cv2.VideoCapture(video_dict['fpath'])
    cap.set(cv2.CAP_PROP_POS_FRAMES,video_dict['start']+display_dict['start']) 

    #set text parameters
    textfont = cv2.FONT_HERSHEY_SIMPLEX
    textposition = (10,30)
    textfontscale = 1
    textlinetype = 2
    textfontcolor = 255

    #Initialize first frame
    ret, frame_new = cap.read()
    frame_new = cv2.cvtColor(frame_new, cv2.COLOR_BGR2GRAY)
    if (video_dict['dsmpl'] < 1):
        frame_new = cv2.resize(
            frame_new,
            (
                int(frame_new.shape[1]*video_dict['dsmpl']),
                int(frame_new.shape[0]*video_dict['dsmpl'])
            ),
            cv2.INTER_NEAREST)
    frame_new = cropframe(frame_new, crop)
    frame_new = cv2.GaussianBlur(frame_new.astype('float'),(0,0),SIGMA)

    #Initialize video storage if desired
    if display_dict['save_video']==True:
        width, height = int(frame_new.shape[1]), int(frame_new.shape[0])
        fourcc = 0
        writer = cv2.VideoWriter(os.path.join(os.path.normpath(video_dict['dpath']), 'video_output.avi'), 
                         fourcc, 20.0, (width, height), isColor=False)

    #Loop through frames to detect frame by frame differences
    for x in range (display_dict['start']+1,display_dict['end']):

        #Attempt to load next frame
        frame_old = frame_new
        ret, frame_new = cap.read()
        if ret == True:
            
            #process frame           
            frame_new = cv2.cvtColor(frame_new, cv2.COLOR_BGR2GRAY)
            if (video_dict['dsmpl'] < 1):
                frame_new = cv2.resize(
                    frame_new,
                    (
                        int(frame_new.shape[1]*video_dict['dsmpl']),
                        int(frame_new.shape[0]*video_dict['dsmpl'])
                    ),
                    cv2.INTER_NEAREST)
            frame_new = cropframe(frame_new, crop)
            frame_new = cv2.GaussianBlur(frame_new.astype('float'),(0,0),SIGMA) 
            frame_dif = np.absolute(frame_new - frame_old)
            frame_cut = (frame_dif > mt_cutoff).astype('uint8')*255

            #Add text to videos, display and save
            texttext = 'FREEZING' if Freezing[x]==100 else 'ACTIVE'
            cv2.putText(frame_new,texttext,textposition,textfont,textfontscale,textfontcolor,textlinetype)
            display = np.concatenate((frame_new.astype('uint8'),frame_cut))
            display_image(display,display_dict['fps'],display_dict['resize'])
            if display_dict['save_video']==True:
                writer.write(display) 

        else: 
            print('No frame detected at frame : ' + str(x) + '.Stopping video play')
            break

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

def PlayVideo_ext(video_dict,display_dict,Freezing,mt_cutoff,crop=None,SIGMA=1):
    """ 
    -------------------------------------------------------------------------------------
    
    Play portion of video back, displaying thresholded video in tandem. Displayed in
    Jupyter Notebook

    -------------------------------------------------------------------------------------
    Args:
        video_dict:: [dict]
            Dictionary with the following keys:
                'dpath' : directory containing files [str]
                'file' : filename with extension, e.g. 'myvideo.wmv' [str]
                'fps' : frames per second of video files to be processed [int]
                'start' : frame at which to start. 0-based [int]
                'end' : frame at which to end.  set to None if processing 
                        whole video [int]
                'dsmpl' : proptional degree to which video should be downsampled
                        by (0-1).
                'ftype' : (only if batch processing) 
                          video file type extension (e.g. 'wmv') [str]
                'FileNames' : (only if batch processing)
                              List of filenames of videos in folder to be batch 
                              processed.  [list]
                
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
                               
        Freezing:: [numpy.array]
            Array defining whether animal is freezing on frame by frame basis.  
            0 = Not Freezing; 100 = Freezing
        
        mt_cutoff:: [float]
            Threshold value for determining magnitude of change sufficient to mark
            pixel as changing from prior frame.
        
        crop:: [holoviews.streams.stream]
            Holoviews stream object enabling dynamic selection in response to 
            cropping tool. `crop.data` contains x and y coordinates of crop
            boundary vertices. Set to None if no cropping supplied.
            
        SIGMA:: [float]
            Sigma value for gaussian filter applied to each image. Passed to 
            OpenCV `cv2.GuassianBlur`.
    
    -------------------------------------------------------------------------------------
    Returns:
        Nothing returned
    
    -------------------------------------------------------------------------------------
    Notes:

    """
    
    #Upoad file
    cap = cv2.VideoCapture(video_dict['fpath'])
    rate = int(1000/display_dict['fps']) #duration each frame is present for, in milliseconds
    cap.set(cv2.CAP_PROP_POS_FRAMES,video_dict['start']+display_dict['start']) 

    #set text parameters
    textfont = cv2.FONT_HERSHEY_SIMPLEX
    textposition = (10,30)
    textfontscale = 1
    textlinetype = 2
    textfontcolor = 255

    #Initialize first frame
    ret, frame_new = cap.read()
    frame_new = cv2.cvtColor(frame_new, cv2.COLOR_BGR2GRAY)
    if (video_dict['dsmpl'] < 1):
        frame_new = cv2.resize(
            frame_new,
            (
                int(frame_new.shape[1]*video_dict['dsmpl']),
                int(frame_new.shape[0]*video_dict['dsmpl'])
            ),
            cv2.INTER_NEAREST)
    frame_new = cropframe(frame_new, crop)
    frame_new = cv2.GaussianBlur(frame_new.astype('float'),(0,0),SIGMA)

    #Initialize video storage if desired
    if display_dict['save_video']==True:
        width, height = int(frame_new.shape[1]), int(frame_new.shape[0])
        fourcc = 0
        writer = cv2.VideoWriter(os.path.join(os.path.normpath(video_dict['dpath']), 'video_output.avi'), 
                         fourcc, 20.0, (width, height), isColor=False)

    #Loop through frames to detect frame by frame differences
    for x in range (display_dict['start']+1,display_dict['end']):

        # press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        #Attempt to load next frame
        frame_old = frame_new
        ret, frame_new = cap.read()
        if ret == True:
            
            #process frame           
            frame_new = cv2.cvtColor(frame_new, cv2.COLOR_BGR2GRAY)
            if (video_dict['dsmpl'] < 1):
                frame_new = cv2.resize(
                    frame_new,
                    (
                        int(frame_new.shape[1]*video_dict['dsmpl']),
                        int(frame_new.shape[0]*video_dict['dsmpl'])
                    ),
                    cv2.INTER_NEAREST)
            frame_new = cropframe(frame_new, crop)
            frame_new = cv2.GaussianBlur(frame_new.astype('float'),(0,0),SIGMA) 
            frame_dif = np.absolute(frame_new - frame_old)
            frame_cut = (frame_dif > mt_cutoff).astype('uint8')*255

            #Add text to videos, display and save
            texttext = 'FREEZING' if Freezing[x]==100 else 'ACTIVE'
            cv2.putText(frame_new,texttext,textposition,textfont,textfontscale,textfontcolor,textlinetype)
            display = np.concatenate((frame_new.astype('uint8'),frame_cut))
            cv2.imshow("preview",display)
            cv2.waitKey(rate)
            if display_dict['save_video']==True:
                writer.write(display) 

        else: 
            print('No frame detected at frame : ' + str(x) + '.Stopping video play')
            break

    #Close video window and video writer if open        
    cv2.destroyAllWindows()
    _=cv2.waitKey(1) 
    if display_dict['save_video']==True:
        writer.release()

        
        
        
            
########################################################################################    
      
def SaveData(video_dict,Motion,Freezing,mt_cutoff,FreezeThresh,MinDuration):
    """ 
    -------------------------------------------------------------------------------------
    
    Saves frame by frame data for motion and freezing to .csv file

    -------------------------------------------------------------------------------------
    Args:
        video_dict:: [dict]
            Dictionary with the following keys:
                'dpath' : directory containing files [str]
                'file' : filename with extension, e.g. 'myvideo.wmv' [str]
                'fps' : frames per second of video files to be processed [int]
                'start' : frame at which to start. 0-based [int]
                'end' : frame at which to end.  set to None if processing 
                        whole video [int]
                'dsmpl' : proptional degree to which video should be downsampled
                        by (0-1).
                'ftype' : (only if batch processing) 
                          video file type extension (e.g. 'wmv') [str]
                'FileNames' : (only if batch processing)
                              List of filenames of videos in folder to be batch 
                              processed.  [list]
                              
        Motion:: [numpy.array]
            Array containing number of pixels per frame whose intensity change from
            previous frame exceeds `mt_cutoff`. Length is number of frames passed to
            function to loop through. Value of first index, corresponding to first frame,
            is set to 0.
        
        Freezing:: [numpy.array]
            Array defining whether animal is freezing on frame by frame basis.  
            0 = Not Freezing; 100 = Freezing
        
        mt_cutoff:: [float]
            Threshold value for determining magnitude of change sufficient to mark
            pixel as changing from prior frame.
                
        FreezeThresh:: [float]
            Threshold value for determining magnitude of activity in `Motion` to designate
            frame as freezing/not freezing.
                
        MinDuration:: [uint8]
            Duration for which `Motion` must be below `FreezeThresh` for freezing to be 
            registered.
    
    -------------------------------------------------------------------------------------
    Returns:
        Nothing returned
        
    -------------------------------------------------------------------------------------
    Notes:


    """

    #Create Dataframe
    DataFrame = pd.DataFrame(
        {'File': [video_dict['file']]*len(Motion),
         'MotionCutoff':np.ones(len(Motion))*mt_cutoff,
         'FreezeThresh':np.ones(len(Motion))*FreezeThresh,
         'MinFreezeDuration':np.ones(len(Motion))*MinDuration,
         'Frame': np.arange(len(Motion)),
         'Motion': Motion,
         'Freezing': Freezing
        })   

    DataFrame.to_csv(os.path.splitext(video_dict['fpath'])[0] + '_FreezingOutput.csv')
    
    
    
    
    
########################################################################################        


def Summarize(video_dict,Motion,Freezing,FreezeThresh,MinDuration,mt_cutoff,bin_dict=None):
    """ 
    -------------------------------------------------------------------------------------
    
    Generate binned summary report of feezing and motion based upon user-specified bins.
    If no bins are specified (bin_dict=None), session average will be returned.

    -------------------------------------------------------------------------------------
    Args:
        video_dict:: [dict]
            Dictionary with the following keys:
                'dpath' : directory containing files [str]
                'file' : filename with extension, e.g. 'myvideo.wmv' [str]
                'fps' : frames per second of video files to be processed [int]
                'start' : frame at which to start. 0-based [int]
                'end' : frame at which to end.  set to None if processing 
                        whole video [int]
                'dsmpl' : proptional degree to which video should be downsampled
                        by (0-1).
                'ftype' : (only if batch processing) 
                          video file type extension (e.g. 'wmv') [str]
                'FileNames' : (only if batch processing)
                              List of filenames of videos in folder to be batch 
                              processed.  [list]
                              
        Motion:: [numpy.array]
            Array containing number of pixels per frame whose intensity change from
            previous frame exceeds `mt_cutoff`. 
        
        Freezing:: [numpy.array]
            Array defining whether animal is freezing on frame by frame basis.  
            0 = Not Freezing; 100 = Freezing
                
        FreezeThresh:: [float]
            Threshold value for determining magnitude of activity in `Motion` to designate
            frame as freezing/not freezing.
                
        MinDuration:: [uint8]
            Duration for which `Motion` must be below `FreezeThresh` for freezing to be 
            registered.
        
        mt_cutoff:: [float]
            Threshold value for determining magnitude of change sufficient to mark
            pixel as changing from prior frame.
        
        bin_dict:: [dict]
            Dictionary specifying bins.  Dictionary keys should be names of the bins.  
            Dictionary value for each bin should be a tuple, with the start and end of 
            the bin, in seconds, relative to the start of the analysis period 
            (i.e. if start frame is 100, it will be relative to that). If no bins are to 
            be specified, set bin_dict = None.
            example = bin_dict = {1:(0,100), 2:(100,200)}
    
    -------------------------------------------------------------------------------------
    Returns:
        df:: [pandas.dataframe]
            Returns pandas dataframe with binned summary information.
        
    -------------------------------------------------------------------------------------
    Notes:


    """
    
    #define bins
    avg_dict = {'all': (0, len(Motion))}
    bin_dict = bin_dict if bin_dict is not None else avg_dict
    #bin_dict = {k: tuple((np.array(v) * video_dict['fps']).tolist()) for k, v in bin_dict.items()}
    
    #get means
    bins = (pd.Series(bin_dict).rename('range(f)')
            .reset_index().rename(columns=dict(index='bin')))
    bins['Motion'] = bins['range(f)'].apply(
        lambda rng: Motion[slice(rng[0],rng[1])].mean())
    bins['Freezing'] = bins['range(f)'].apply(
        lambda rng: Freezing[slice(rng[0],rng[1])].mean())
    
    #Create data frame to store data in
    df = pd.DataFrame({
        'File': [video_dict['file']]*len(bins),
        'FileLength': np.ones(len(bins))*len(Motion),
        'MotionCutoff':np.ones(len(bins))*mt_cutoff,
        'FreezeThresh':np.ones(len(bins))*FreezeThresh,
        'MinFreezeDuration':np.ones(len(bins))*MinDuration
    })   
    df = pd.concat([df,bins],axis=1)
    return df





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
                'fps' : frames per second of video files to be processed [int]
                'start' : frame at which to start. 0-based [int]
                'end' : frame at which to end.  set to None if processing 
                        whole video [int]
                'dsmpl' : proptional degree to which video should be downsampled
                        by (0-1).
                'ftype' : (only if batch processing) 
                          video file type extension (e.g. 'wmv') [str]
                'FileNames' : (only if batch processing)
                              List of filenames of videos in folder to be batch 
                              processed.  [list]

    
    -------------------------------------------------------------------------------------
    Returns:
        video_dict:: [dict]
            Dictionary with the following keys:
                'dpath' : directory containing files [str]
                'file' : filename with extension, e.g. 'myvideo.wmv' [str]
                'fps' : frames per second of video file/files to be processed [int]
                'start' : frame at which to start. 0-based [int]
                'end' : frame at which to end.  set to None if processing whole 
                        video [int]
                'ftype' : (only if batch processing) 
                          video file type extension (e.g. 'wmv') [str]
                'FileNames' : (only if batch processing)
                              List of filenames of videos in folder to be 
                              batch processed.  [list]
    
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
        
        
def Batch(video_dict,bin_dict,mt_cutoff,FreezeThresh,MinDuration,crop=None,SIGMA=1):
    """ 
    -------------------------------------------------------------------------------------
    
    Run FreezeAnalysis on folder of videos of specified filetype. 
    
    -------------------------------------------------------------------------------------
    Args:
        video_dict:: [dict]
            Dictionary with the following keys:
                'dpath' : directory containing files [str]
                'file' : filename with extension, e.g. 'myvideo.wmv' [str]
                'fps' : frames per second of video files to be processed [int]
                'start' : frame at which to start. 0-based [int]
                'end' : frame at which to end.  set to None if processing 
                        whole video [int]
                'dsmpl' : proptional degree to which video should be downsampled
                        by (0-1).
                'ftype' : (only if batch processing) 
                          video file type extension (e.g. 'wmv') [str]
                'FileNames' : (only if batch processing)
                              List of filenames of videos in folder to be batch 
                              processed.  [list]
                              
        bin_dict:: [dict]
            Dictionary specifying bins.  Dictionary keys should be names of the bins.  
            Dictionary value for each bin should be a tuple, with the start and end of 
            the bin, in seconds, relative to the start of the analysis period 
            (i.e. if start frame is 100, it will be relative to that). If no bins are to 
            be specified, set bin_dict = None.
            example = bin_dict = {1:(0,100), 2:(100,200)}
        
        mt_cutoff:: [float]
            Threshold value for determining magnitude of change sufficient to mark
            pixel as changing from prior frame.
        
        FreezeThresh:: [float]
            Threshold value for determining magnitude of activity in `Motion` to designate
            frame as freezing/not freezing.
                
        MinDuration:: [uint8]
            Duration for which `Motion` must be below `FreezeThresh` for freezing to be 
            registered.
            
        crop:: [holoviews.streams.stream]
            Holoviews stream object enabling dynamic selection in response to 
            cropping tool. `crop.data` contains x and y coordinates of crop
            boundary vertices. Set to None if no cropping supplied.
            
        SIGMA:: [float]
            Sigma value for gaussian filter applied to each image. Passed to 
            OpenCV `cv2.GuassianBlur`    

    
    -------------------------------------------------------------------------------------
    Returns:
        summary_all:: [pandas.dataframe]
            Pandas dataframe with binned summary information for all videos
            processed. If `bin_dict = None`, average freezing and motion for each video
            will be returned.
    
    -------------------------------------------------------------------------------------
    Notes:
    
    """

    #Loop through files    
    for video_dict['file'] in video_dict['FileNames']:

        #Set file
        print ('Processing File: {f}'.format(f=video_dict['file']))
        video_dict['fpath'] = os.path.join(os.path.normpath(video_dict['dpath']), video_dict['file'])

        #Analyze frame by frame motion and freezing and save csv of results
        Motion = Measure_Motion(video_dict,mt_cutoff,crop,SIGMA=1)  
        Freezing = Measure_Freezing(Motion,FreezeThresh,MinDuration)  
        SaveData(video_dict,Motion,Freezing,mt_cutoff,FreezeThresh,MinDuration)
        summary = Summarize(video_dict,Motion,Freezing,FreezeThresh,
                            MinDuration,mt_cutoff,bin_dict=bin_dict)

        #Add summary info for individual file to larger summary of all files
        try:
            summary_all = pd.concat([summary_all,summary])
        except NameError: #to be done for first file in list, before summary_all is created
            summary_all = summary

    #Write summary data to csv file
    sum_pathout = os.path.join(os.path.normpath(video_dict['dpath']), 'BatchSummary.csv')
    summary_all.to_csv(sum_pathout)
    return summary_all





########################################################################################

def Calibrate(video_dict,cal_pix,SIGMA):
    """ 
    -------------------------------------------------------------------------------------
    
    Using empty video (i.e. no animal), find distribution of frame by frame pixel changes.
    99.99 percentile is printed, and twice this number is recommended threshold for
    `mt_cutoff`. Additionally, histogram of distribution is returned.
    
    -------------------------------------------------------------------------------------
    Args:
        video_dict:: [dict]
            Dictionary with the following keys:
                'dpath' : directory containing files [str]
                'file' : filename with extension, e.g. 'myvideo.wmv' [str]
                'fps' : frames per second of video files to be processed [int]
                'cal_sec' : number of seconds to calibrate based upon [int]
                'dsmpl' : proptional degree to which video should be downsampled
                        by (0-1).
        
        cal_pix:: [int]
            Number of pixels in frame to base calibration upon. Random selection of 
            all pixels, sampled with replacement. Note that sampling strategy was
            implemented to reduce memory load.
            
        SIGMA:: [float]
            Sigma value for gaussian filter applied to each image. Passed to 
            OpenCV `cv2.GuassianBlur`    

    
    -------------------------------------------------------------------------------------
    Returns:
        hist*vline:: [holoviews.Overlay]
            Holoviews Overlay of hv.Histogram and hv.VLine. Histogram displays
            distribution of frame by frame intensity changes and vertical line displays
            suggested motion cutoff.
            
    -------------------------------------------------------------------------------------
    Notes:
    
    """
    
    #Upoad file
    cap = cv2.VideoCapture(video_dict['fpath'])
    
    #set seconds to examine and frames
    cal_frames = video_dict['cal_sec']*video_dict['fps']

    #Initialize matrix for difference values
    cal_dif = np.zeros((cal_frames,cal_pix))

    #Initialize video
    cap.set(cv2.CAP_PROP_POS_FRAMES,0) 

    #Initialize first frame
    ret, frame = cap.read()
    frame_new = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if (video_dict['dsmpl'] < 1):
        frame_new = cv2.resize(
            frame_new,
            (
                int(frame_new.shape[1]*video_dict['dsmpl']),
                int(frame_new.shape[0]*video_dict['dsmpl'])
            ),
            cv2.INTER_NEAREST)
    frame_new = cv2.GaussianBlur(frame_new.astype('float'),(0,0),SIGMA)

    #Get random set of pixels to examine across frames
    h,w=frame_new.shape
    h_loc = np.random.rand(cal_pix,1)*h
    h_loc = h_loc.astype(int)
    w_loc = np.random.rand(cal_pix,1)*w
    w_loc = w_loc.astype(int)

    #Loop through frames to detect frame by frame differences
    for x in range (1,cal_frames):

        #Reset old frame
        frame_old = frame_new

        #Load next frame
        ret, frame = cap.read()
        
        if ret == True:
            #Process frame
            frame_new = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if (video_dict['dsmpl'] < 1):
                frame_new = cv2.resize(
                    frame_new,
                    (
                        int(frame_new.shape[1]*video_dict['dsmpl']),
                        int(frame_new.shape[0]*video_dict['dsmpl'])
                    ),
                    cv2.INTER_NEAREST)
            frame_new = cv2.GaussianBlur(frame_new.astype('float'),(0,0),SIGMA)

            #Get differences for select pixels
            frame_pix_dif = np.absolute(frame_new[h_loc,w_loc] - frame_old[h_loc,w_loc])
            frame_pix_dif = frame_pix_dif[:,0]

            #Populate difference array
            cal_dif[x,:]=frame_pix_dif
            
        else: #if no frame returned
            cal_dif = cal_dif[:x,:]
            print('Only {a} frames detected'.format(a=x))
            break
    
    percentile = np.percentile(cal_dif,99.99)

    #Calculate grayscale change cutoff for detecting motion
    cal_dif_avg = np.nanmean(cal_dif)

    #Set Cutoff
    mt_cutoff = 2*percentile

    #Print stats and selected cutoff
    print ('Average frame-by-frame pixel difference: ' + str(cal_dif_avg))
    print ('99.99 percentile of pixel change differences: ' + str(percentile))
    print ('Grayscale change cut-off for pixel change: ' + str(mt_cutoff))
    
    hist_freqs, hist_edges = np.histogram(cal_dif,bins=np.arange(30),density=True)
    hist = hv.Histogram((hist_edges, hist_freqs))
    hist.opts(title="Motion Cutoff: "+str(np.around(mt_cutoff,1)),xlabel="Grayscale Change",ylabel=
             'Proportion')
    vline = hv.VLine(mt_cutoff)
    vline.opts(color='red')
    return hist*vline





########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
#OLD STUFF
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################


# def Measure_Motion_OLD (video_dict,crop,mt_cutoff,SIGMA):
    
#     #Extract ycrop value 
#     try: #if passed as x,y coordinates
#         if len(crop.data['y'])!=0:
#             ycrop = int(np.around(crop.data['y'][0]))
#         else:
#             ycrop=0
#     except: #if passed as single value
#         ycrop=crop
    
#     #Upoad file
#     cap = cv2.VideoCapture(video_dict['fpath'])
#     cap_max = int(cap.get(7)) #7 is index of total frames
#     cap_max = int(video_dict['end']) if video_dict['end'] is not None else cap_max
#     cap.set(1,video_dict['start']) #first index references frame property, second specifies next frame to grab

#     #Initialize first frame
#     ret, frame = cap.read()
#     frame_new = cv2.GaussianBlur(frame[ycrop:,:].astype('float'),(0,0),SIGMA)
    
#     #Initialize vector to store motion values in
#     Motion = np.zeros(cap_max - video_dict['start'])

#     #Loop through frames to detect frame by frame differences
#     for x in range (1,len(Motion)):
#         frame_old = frame_new
#         ret, frame = cap.read()
#         if ret == True:
#             #Reset new frame and process calculate difference between old/new frames
#             #frame_new = mh.gaussian_filter(frame[ycrop:,:],sigma=SIGMA) 
#             frame_new = cv2.GaussianBlur(frame[ycrop:,:].astype('float'),(0,0),SIGMA)
#             frame_dif = np.absolute(frame_new - frame_old)
#             frame_cut = (frame_dif > mt_cutoff).astype('uint8')
#             Motion[x]=np.sum(frame_cut)
#         else: 
#             #if no frame is detected
#             x = x-1 #Reset x to last frame detected
#             Motion = Motion[:x] #Amend length of motion vector
#             break
        
#     cap.release() #release video
#     return(Motion) #return motion values


########################################################################################        

# def LoadAndCrop_OLD (video_dict,stretch={'width':1,'height':1},cropmethod=None,batch=False):

#     #if batch processing, set file to first file to be processed
#     if batch:
#         video_dict['file'] = video_dict['FileNames'][0]
        
#     #Upoad file and check that it exists
#     video_dict['fpath'] = os.path.join(os.path.normpath(video_dict['dpath']), video_dict['file'])
#     if os.path.isfile(video_dict['fpath']):
#         print('file: {file}'.format(file=video_dict['fpath']))
#         cap = cv2.VideoCapture(video_dict['fpath'])
#     else:
#         raise FileNotFoundError('{file} not found. Check that directory and file names are correct'.format(
#             file=video_dict['fpath']))

#     #Get maxiumum frame of file. Note that max frame is updated later if fewer frames detected
#     cap_max = int(cap.get(7)) #7 is index of total frames
#     print('total frames: {frames}'.format(frames=cap_max))

#     #Set first frame. 
#     try:
#         cap.set(1,video_dict['start']) #first index references frame property, second specifies next frame to grab
#     except:
#         cap.set(1,0)
#     ret, frame = cap.read() 
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   
#     cap.release() 

#     #Make first image reference frame on which cropping can be performed
#     image = hv.Image((np.arange(frame.shape[1]), np.arange(frame.shape[0]), frame))
#     image.opts(width=int(frame.shape[1]*stretch['width']),
#                height=int(frame.shape[0]*stretch['height']),
#               invert_yaxis=True,cmap='gray',
#               colorbar=True,
#                toolbar='below',
#               title="First Frame.  Crop if Desired")
    
#     #Create polygon element on which to draw and connect via stream to poly drawing tool
#     if cropmethod==None:
#         image.opts(title="First Frame")
#         return image,None,video_dict
    
#     if cropmethod=='Box':         
#         box = hv.Polygons([])
#         box.opts(alpha=.5)
#         box_stream = streams.BoxEdit(source=box,num_objects=1)     
#         return (image*box),box_stream,video_dict
    
#     if cropmethod=='HLine':  
#         points = hv.Points([])
#         points.opts(active_tools=['point_draw'], color='white',size=1)
#         pointerXY_stream = streams.PointerXY(x=0, y=0, source=image)
#         pointDraw_stream = streams.PointDraw(source=points,num_objects=1)
            
#         def h_track(x, y): #function to track pointer
#             y = int(np.around(y))
#             text = hv.Text(x, y, str(y), halign='left', valign='bottom')
#             return hv.HLine(y) * text
#         track=hv.DynamicMap(h_track, streams=[pointerXY_stream])
        
#         def h_line(data): #function to draw line
#             try:
#                 hline=hv.HLine(data['y'][0])
#                 return hline
#             except:
#                 hline=hv.HLine(0)
#                 return hline
#         line=hv.DynamicMap(h_line,streams=[pointDraw_stream])
        
#         def h_text(data): #function to write ycrop value
#             center=frame.shape[1]//2 
#             try:
#                 y=int(np.around(data['y'][0]))
#                 htext=hv.Text(center,y+10,'ycrop: {x}'.format(x=y))
#                 return htext
#             except:
#                 htext=hv.Text(center,10, 'ycrop: 0')
#                 return htext
#         text=hv.DynamicMap(h_text,streams=[pointDraw_stream])
        
        
#         return image*track*points*line*text,pointDraw_stream,video_dict   

