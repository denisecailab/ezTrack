#
#List of Functions in FreezeAnalysis_Functions
#

# Check -
# LoadAndCrop - 
# Measure_Motion -
# Measure_Freezing -
# Play_Video -
# Save_Data -
# Summarize -
# Batch -
# Calibrate -

########################################################################################

import os
import sys
import cv2
import fnmatch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from scipy import ndimage
import holoviews as hv
from holoviews import opts
from holoviews import streams
from holoviews.streams import Stream, param
hv.notebook_extension('bokeh')
warnings.filterwarnings("ignore")


########################################################################################        

def LoadAndCrop(video_dict,stretch={'width':1,'height':1},cropmethod='none'):

    #if batch processing, set file to first file to be processed
    try:
        video_dict['file'] = video_dict['FileNames'][0]
    except:
        video_dict['file'] = video_dict['file']
    
    #Upoad file and check that it exists
    video_dict['fpath'] = os.path.join(os.path.normpath(video_dict['dpath']), video_dict['file'])
    if os.path.isfile(video_dict['fpath']):
        print('file: {file}'.format(file=video_dict['fpath']))
        cap = cv2.VideoCapture(video_dict['fpath'])
    else:
        raise FileNotFoundError('{file} not found. Check that directory and file names are correct'.format(
            file=video_dict['fpath']))

    #Get maxiumum frame of file. Note that max frame is updated later if fewer frames detected
    cap_max = int(cap.get(7)) #7 is index of total frames
    print('total frames: {frames}'.format(frames=cap_max))

    #Set first frame. 
    try:
        cap.set(1,video_dict['start']) #first index references frame property, second specifies next frame to grab
    except:
        cap.set(1,0)
    ret, frame = cap.read() 
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   
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
    if cropmethod=='none':
        image.opts(title="First Frame")
        return image,None,video_dict
    
    if cropmethod=='Box':         
        box = hv.Polygons([])
        box.opts(alpha=.5)
        box_stream = streams.BoxEdit(source=box,num_objects=1)     
        return (image*box),box_stream,video_dict
    
    if cropmethod=='HLine':  
        points = hv.Points([])
        points.opts(active_tools=['point_draw'], color='white',size=1)
        pointerXY_stream = streams.PointerXY(x=0, y=0, source=image)
        pointDraw_stream = streams.PointDraw(source=points,num_objects=1)
            
        def h_track(x, y): #function to track pointer
            y = int(np.around(y))
            text = hv.Text(x, y, str(y), halign='left', valign='bottom')
            return hv.HLine(y) * text
        track=hv.DynamicMap(h_track, streams=[pointerXY_stream])
        
        def h_line(data): #function to draw line
            try:
                hline=hv.HLine(data['y'][0])
                return hline
            except:
                hline=hv.HLine(0)
                return hline
        line=hv.DynamicMap(h_line,streams=[pointDraw_stream])
        
        def h_text(data): #function to write ycrop value
            center=frame.shape[1]//2 
            try:
                y=int(np.around(data['y'][0]))
                htext=hv.Text(center,y+10,'ycrop: {x}'.format(x=y))
                return htext
            except:
                htext=hv.Text(center,10, 'ycrop: 0')
                return htext
        text=hv.DynamicMap(h_text,streams=[pointDraw_stream])
        
        return image*track*points*line*text,pointDraw_stream,video_dict   
    

########################################################################################

def Measure_Motion(video_dict,crop,mt_cutoff,SIGMA):
    
    #Extract ycrop value 
    try: #if passed as x,y coordinates
        if len(crop.data['y'])!=0:
            ycrop = int(np.around(crop.data['y'][0]))
        else:
            ycrop=0
    except: #if passed as single value
        ycrop=crop
    
    #Upoad file
    cap = cv2.VideoCapture(video_dict['fpath'])
    cap_max = int(cap.get(7)) #7 is index of total frames
    cap_max = int(video_dict['end']) if video_dict['end'] is not None else cap_max
    cap.set(1,video_dict['start']) #first index references frame property, second specifies next frame to grab

    #Initialize first frame
    ret, frame = cap.read()
    frame_new = cv2.GaussianBlur(frame[ycrop:,:].astype('float'),(0,0),SIGMA)
    
    #Initialize vector to store motion values in
    Motion = np.zeros(cap_max - video_dict['start'])

    #Loop through frames to detect frame by frame differences
    for x in range (1,len(Motion)):
        frame_old = frame_new
        ret, frame = cap.read()
        if ret == True:
            #Reset new frame and process calculate difference between old/new frames
            #frame_new = mh.gaussian_filter(frame[ycrop:,:],sigma=SIGMA) 
            frame_new = cv2.GaussianBlur(frame[ycrop:,:].astype('float'),(0,0),SIGMA)
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

def Measure_Freezing(Motion,FreezeThresh,MinDuration):

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

def PlayVideo(video_dict,display_dict,Freezing,mt_cutoff,crop,SIGMA):
    
    #Extract ycrop value 
    try: #if passed as x,y coordinates
        if len(crop.data['y'])!=0:
            ycrop = int(np.around(crop.data['y'][0]))
        else:
            ycrop=0
    except: #if passed as single value
        ycrop=crop
    
    #Upoad file
    cap = cv2.VideoCapture(video_dict['fpath'])
    rate = int(1000/video_dict['fps']) #duration each frame is present for, in milliseconds
    cap.set(1,video_dict['start']+display_dict['start']) #set reference position of first frame 

    #set text parameters
    textfont = cv2.FONT_HERSHEY_SIMPLEX
    textposition = (10,30)
    textfontscale = 1
    textlinetype = 2
    textfontcolor = 255

    #Initialize first frame
    ret, frame = cap.read()
    frame_new = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_new = frame_new[ycrop:,:]
    frame_new = cv2.GaussianBlur(frame_new.astype('float'),(0,0),SIGMA)

    #Initialize video storage if desired
    if display_dict['save_video']==True:
        width = int(frame.shape[1])
        height = int(frame.shape[0]+frame.shape[0]-ycrop)
        fourcc = 0#cv2.VideoWriter_fourcc(*'jpeg') #only writes up to 20 fps, though video read can be 30.
        #fourcc = cv2.VideoWriter_fourcc(*'MJPG') #only writes up to 20 fps, though video read can be 30.
        writer = cv2.VideoWriter(os.path.join(os.path.normpath(video_dict['dpath']), 'video_output.avi'), 
                         fourcc, 20.0, 
                         (width, height),
                         isColor=False)

    #Loop through frames to detect frame by frame differences
    for x in range (display_dict['start']+1,display_dict['end']):

        # press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        #Attempt to load next frame
        ret, frame = cap.read()
        if ret == True:

            #Convert to gray scale
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #set old frame to new frame and get new frame
            frame_old = frame_new
            frame_new = cv2.GaussianBlur(frame[ycrop:,:].astype('float'),(0,0),SIGMA)
            #Calculate difference between frames
            frame_dif = np.absolute(frame_new - frame_old)
            frame_cut = (frame_dif > mt_cutoff).astype('uint8')*255
            #Add text to videos
            texttext = 'FREEZING' if Freezing[x]==100 else 'ACTIVE'
            cv2.putText(frame,texttext,textposition,textfont,textfontscale,textfontcolor,textlinetype)
            #Display video
            display = np.concatenate((frame,frame_cut))
            cv2.imshow("preview",display)
            cv2.waitKey(rate)
            #Save video (if desired). 
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

    #Create Dataframe
    DataFrame = pd.DataFrame(
        {'File': [video_dict['file']]*len(Motion),
         'FPS': np.ones(len(Motion))*video_dict['fps'],
         'MotionCutoff':np.ones(len(Motion))*mt_cutoff,
         'FreezeThresh':np.ones(len(Motion))*FreezeThresh,
         'MinFreezeDuration':np.ones(len(Motion))*MinDuration,
         'Frame': np.arange(len(Motion)),
         'Motion': Motion,
         'Freezing': Freezing
        })   

    DataFrame.to_csv(os.path.splitext(video_dict['fpath'])[0] + '_FreezingOutput.csv')
    
########################################################################################        


def Summarize(video_dict,Motion,Freezing,FreezeThresh,MinDuration,crop,mt_cutoff,bin_dict=None):
    
    #define bins
    avg_dict = {'all': (0, len(Motion))}
    try:
        bin_dict = {k: tuple((np.array(v) * video_dict['fps']).tolist()) for k, v in bin_dict.items()}
    except AttributeError:
        bin_dict = avg_dict 
    
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
        'FPS': np.ones(len(bins))*video_dict['fps'],
        'MotionCutoff':np.ones(len(bins))*mt_cutoff,
        'FreezeThresh':np.ones(len(bins))*FreezeThresh,
        'MinFreezeDuration':np.ones(len(bins))*MinDuration
    })   
    df = pd.concat([df,bins],axis=1)
    return df



########################################################################################

def Batch(video_dict,bin_dict,crop,mt_cutoff,FreezeThresh,MinDuration,SIGMA=1):
    
    #Get list of video files
    if os.path.isdir(video_dict['dpath']):
        video_dict['FileNames'] = sorted(os.listdir(video_dict['dpath']))
        video_dict['FileNames'] = fnmatch.filter(video_dict['FileNames'], ('*.' + video_dict['ftype'])) 
    else:
        raise FileNotFoundError('{path} not found. Check that directory is correct'.format(
            path=video_dict['dpath']))

    #Loop through files    
    for video_dict['file'] in video_dict['FileNames']:

        #Set file
        print ('Processing File: {f}'.format(f=video_dict['file']))
        video_dict['fpath'] = os.path.join(os.path.normpath(video_dict['dpath']), video_dict['file'])

        #Analyze frame by frame motion and freezing and save csv of results
        Motion = Measure_Motion(video_dict,crop,mt_cutoff,SIGMA=1)  
        Freezing = Measure_Freezing(Motion,FreezeThresh,MinDuration)  
        SaveData(video_dict,Motion,Freezing,mt_cutoff,FreezeThresh,MinDuration)
        summary = Summarize(video_dict,Motion,Freezing,FreezeThresh,
                            MinDuration,crop,mt_cutoff,bin_dict=bin_dict)

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
    
    #Upoad file
    cap = cv2.VideoCapture(video_dict['fpath'])
    
    #set seconds to examine and frames
    cal_frames = video_dict['cal_sec']*video_dict['fps']

    #Initialize matrix for difference values
    cal_dif = np.zeros((cal_frames,cal_pix))

    #Initialize video
    cap.set(1,0) #first index references frame property, second specifies next frame to grab

    #Initialize first frame
    ret, frame = cap.read()
    frame_new = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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
            frame_new = cv2.GaussianBlur(frame_new.astype('float'),(0,0),SIGMA)
            #frame_new = mh.gaussian_filter(frame_new,sigma=SIGMA) # used to reduce influence of jitter from one frame to the next

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
       
