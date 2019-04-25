#
#List of Functions in LocationTracking_Functions.py
#

# LoadAndCrop - 
# Reference -
# Locate -
# TrackLocation -
# LocationThresh_View -
# ROI_plot -
# ROI_Location -
# Batch_LoadFiles - 
# Batch_Process -
# Play Video -

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
    if 'file' not in video_dict.keys():
        video_dict['file'] = video_dict['FileNames'][0]   
    
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

    #Set first frame
    cap.set(1,video_dict['start']) #first index references frame property, second specifies next frame to grab
    ret, frame = cap.read() 
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   
    cap.release()
    print('dimensions: {x}'.format(x=frame.shape))

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
    
def Reference(video_dict,crop,num_frames=100):
    
    #if batch processing, set file to first file to be processed
    if 'file' not in video_dict.keys():
        video_dict['file'] = video_dict['FileNames'][0]        
    
    #get correct ref video
    vname = video_dict.get("altfile", video_dict['file'])
    fpath = os.path.join(os.path.normpath(video_dict['dpath']), vname)
    if os.path.isfile(fpath):
        cap = cv2.VideoCapture(fpath)
    else:
        raise FileNotFoundError('File not found. Check that directory and file names are correct.')

    #Upoad file
    cap.set(1,0)#first index references frame property, second specifies next frame to grab
    
    #Get video dimensions with any cropping applied
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    try: #if frame is cropped
        Xs=[crop.data['x0'][0],crop.data['x1'][0]]
        Ys=[crop.data['y0'][0],crop.data['y1'][0]]
        fxmin,fxmax=int(min(Xs)), int(max(Xs))
        fymin,fymax=int(min(Ys)), int(max(Ys))
    except: #if no crop
        fxmin,fxmax=0,frame.shape[1]
        fymin,fymax=0,frame.shape[0]
    h,w=(fymax-fymin),(fxmax-fxmin)
    cap_max = int(cap.get(7)) #7 is index of total frames
    cap_max = int(video_dict['end']) if video_dict['end'] is not None else cap_max
    
    #Collect subset of frames
    collection = np.zeros((num_frames,h,w))  
    for x in range (num_frames):          
        grabbed = False
        while grabbed == False: 
            y=np.random.randint(video_dict['start'],cap_max)
            cap.set(1,y)#first index references frame property, second specifies next frame to grab
            ret, frame = cap.read()
            if ret == True:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = gray[fymin:fymax,fxmin:fxmax]
                collection[x,:,:]=gray
                grabbed = True
            elif ret == False:
                pass
    cap.release() 
    
    reference = np.median(collection,axis=0)
    return reference    

########################################################################################

def Locate(cap,crop,reference,tracking_params,prior=None):    
    
    #attempt to load frame
    ret, frame = cap.read() #read frame
    
    #set window dimensions
    if prior != None and tracking_params['use_window']==True:
        window_size = tracking_params['window_size']//2
        ymin,ymax = prior[0]-window_size, prior[0]+window_size
        xmin,xmax = prior[1]-window_size, prior[1]+window_size

    if ret == True:
        
        #load frame and crop
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        try: #if crop is supplied
            Xs=[crop.data['x0'][0],crop.data['x1'][0]]
            Ys=[crop.data['y0'][0],crop.data['y1'][0]]
            fxmin,fxmax=int(min(Xs)), int(max(Xs))
            fymin,fymax=int(min(Ys)), int(max(Ys))
        except: #if no cropping is used
            fxmin,fxmax=0,frame.shape[1]
            fymin,fymax=0,frame.shape[0]
        frame = frame[fymin:fymax,fxmin:fxmax]
        
        #find difference from reference and blur
        dif = np.absolute(frame-reference)
        dif = dif.astype('uint8')
              
        #apply window
        weight = 1 - tracking_params['window_weight']
        if prior != None and tracking_params['use_window']==True:
            dif_weights = np.ones(dif.shape)*weight
            dif_weights[slice(ymin if ymin>0 else 0, ymax),
                        slice(xmin if xmin>0 else 0, xmax)]=1
            dif = dif*dif_weights
            
        #threshold differences and find center of mass for remaining values
        dif[dif<np.percentile(dif,tracking_params['loc_thresh'])]=0
        com=ndimage.measurements.center_of_mass(dif)
        return ret, dif, com, frame
    
    else:
        return ret, None, None, frame
        
########################################################################################        

def TrackLocation(video_dict,tracking_params,reference,crop):
          
    #load video
    cap = cv2.VideoCapture(video_dict['fpath'])#set file
    cap.set(1,video_dict['start']) #set starting frame
    cap_max = int(cap.get(7)) #get max frames. 7 is index of total frames
    cap_max = int(video_dict['end']) if video_dict['end'] is not None else cap_max
   
    
    #Initialize vector to store motion values in
    X = np.zeros(cap_max - video_dict['start'])
    Y = np.zeros(cap_max - video_dict['start'])
    D = np.zeros(cap_max - video_dict['start'])

    #Loop through frames to detect frame by frame differences
    for f in range(len(D)):
        
        if f>0: 
            yprior = np.around(Y[f-1]).astype(int)
            xprior = np.around(X[f-1]).astype(int)
            ret,dif,com,frame = Locate(cap,crop,reference,tracking_params,prior=[yprior,xprior])
        else:
            ret,dif,com,frame = Locate(cap,crop,reference,tracking_params)
                                                
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
    print('total frames processed: {f}'.format(f=len(D)))
    
    #create pandas dataframe
    df = pd.DataFrame(
    {'File' : video_dict['file'],
     'FPS': np.ones(len(D))*video_dict['fps'],
     'Location_Thresh': np.ones(len(D))*tracking_params['loc_thresh'],
     'Use_Window': str(tracking_params['use_window']),
     'Window_Weight': np.ones(len(D))*tracking_params['window_weight'],
     'Window_Size': np.ones(len(D))*tracking_params['window_size'],
     'Start_Frame': np.ones(len(D))*video_dict['start'],
     'Frame': np.arange(len(D)),
     'X': X,
     'Y': Y,
     'Distance': D
    })
       
    return df


########################################################################################

def LocationThresh_View(examples,video_dict,reference,crop,tracking_params,stretch):
    
    #load video
    cap = cv2.VideoCapture(video_dict['fpath'])
    cap_max = int(cap.get(7)) #get max frames. 7 is index of total frames
    cap_max = int(video_dict['end']) if video_dict['end'] is not None else cap_max
    
    #examine random frames
    images = []
    for example in range (examples):
        
        #analyze frame
        frm=np.random.randint(video_dict['start'],cap_max) #select random frame
        cap.set(1,frm) #sets frame to be next to be grabbed
        ret,dif,com,frame = Locate(cap,crop,reference,tracking_params) #get frame difference from reference 

        #plot original frame
        image_orig = hv.Image((np.arange(frame.shape[1]), np.arange(frame.shape[0]), frame))
        image_orig.opts(width=int(reference.shape[1]*stretch['width']),
                   height=int(reference.shape[0]*stretch['height']),
                   invert_yaxis=True,cmap='gray',toolbar='below',
                   title="Frame: " + str(frm))
        orig_overlay = image_orig * hv.Points(([com[1]],[com[0]])).opts(
            color='red',size=20,marker='+',line_width=3) 
        
        #plot heatmap
        dif = dif*(255//dif.max())
        image_heat = hv.Image((np.arange(dif.shape[1]), np.arange(dif.shape[0]), dif))
        image_heat.opts(width=int(dif.shape[1]*stretch['width']),
                   height=int(dif.shape[0]*stretch['height']),
                   invert_yaxis=True,cmap='jet',toolbar='below',
                   title="Frame: " + str(frm))
        heat_overlay = image_heat * hv.Points(([com[1]],[com[0]])).opts(
            color='red',size=20,marker='+',line_width=3) 
        
        images.extend([orig_overlay,heat_overlay])
    
    cap.release()
    layout = hv.Layout(images)
    return layout

########################################################################################    
    
def ROI_plot(reference,region_names,stretch):
    
    #Define parameters for plot presentation
    nobjects = len(region_names) #get number of objects to be drawn

    #Make reference image the base image on which to draw
    image = hv.Image((np.arange(reference.shape[1]), np.arange(reference.shape[0]), reference))
    image.opts(width=int(reference.shape[1]*stretch['width']),
               height=int(reference.shape[0]*stretch['height']),
              invert_yaxis=True,cmap='gray',
              colorbar=True,
               toolbar='above',
              title="Draw Regions: "+', '.join(region_names))

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
        rois = region_names[:len(xs)]
        return hv.Labels((xs, ys, rois))

    dmap = hv.DynamicMap(centers, streams=[poly_stream])
    
    return (image * poly * dmap), poly_stream

########################################################################################    

def ROI_Location(reference,poly_stream,region_names,location):

    #Create ROI Masks
    ROI_masks = {}
    for poly in range(len(poly_stream.data['xs'])):
        x = np.array(poly_stream.data['xs'][poly]) #x coordinates
        y = np.array(poly_stream.data['ys'][poly]) #y coordinates
        xy = np.column_stack((x,y)).astype('uint64') #xy coordinate pairs
        mask = np.zeros(reference.shape) # create empty mask
        cv2.fillPoly(mask, pts =[xy], color=255) #fill polygon  
        ROI_masks[region_names[poly]] = mask==255 #save to ROI masks as boolean 

    #Create arrays to store whether animal is within given ROI
    ROI_location = {}
    for mask in ROI_masks:
        ROI_location[mask]=np.full(len(location['Frame']),False,dtype=bool)

    #For each frame assess truth of animal being in each ROI
    for f in location['Frame']:
        y,x = location['Y'][f], location['X'][f]
        for mask in ROI_masks:
            ROI_location[mask][f] = ROI_masks[mask][int(y),int(x)]
    
    #Add date to location data frame
    for x in ROI_location:
        location[x]=ROI_location[x]
    
    return location
    
########################################################################################        
    
def Summarize_Location(location, video_dict, bin_dict=None, region_names=None):
    
    avg_dict = {'all': (location['Frame'].min(), location['Frame'].max())}
    
    try:
        bin_dict = {k: tuple((np.array(v) * video_dict['fps']).tolist()) for k, v in bin_dict.items()}
    except AttributeError:
        bin_dict = avg_dict
    bins = (pd.Series(bin_dict).rename('range(f)')
            .reset_index().rename(columns=dict(index='bin')))
    
    bins['Distance'] = bins['range(f)'].apply(
        lambda r: location[location['Frame'].between(*r)]['Distance'].sum())
    if region_names is not None:
        bins_reg = bins['range(f)'].apply(
            lambda r: location[location['Frame'].between(*r)][region_names].mean())
        bins = bins.join(bins_reg)
        drp_cols = ['Distance', 'Frame', 'X', 'Y'] + region_names
    else:
        drp_cols = ['Distance', 'Frame', 'X', 'Y']
    bins = pd.merge(
        location.drop(drp_cols, axis='columns'),
        bins,
        left_index=True,
        right_index=True)
    
    return bins

######################################################################################## 

def Batch_LoadFiles(video_dict):

    #Get list of video files of designated type
    if os.path.isdir(video_dict['dpath']):
        video_dict['FileNames'] = sorted(os.listdir(video_dict['dpath']))
        video_dict['FileNames'] = fnmatch.filter(video_dict['FileNames'], ('*.' + video_dict['ftype'])) 
        crop,poly_stream = None,None
        return video_dict, crop, poly_stream
    else:
        raise FileNotFoundError('{path} not found. Check that directory is correct'.format(
            path=video_dict['dpath']))

######################################################################################## 

def Batch_Process(video_dict,tracking_params,bin_dict,region_names,stretch,crop,poly_stream=None):
    
    #get polygon
    if poly_stream != None:
        lst = []
        for poly in range(len(poly_stream.data['xs'])):
            x = np.array(poly_stream.data['xs'][poly]) #x coordinates
            y = np.array(poly_stream.data['ys'][poly]) #y coordinates
            lst.append( [ (x[vert],y[vert]) for vert in range(len(x)) ] )
        poly = hv.Polygons(lst).opts(fill_alpha=0.1,line_dash='dashed')
    
    heatmaps = []
    for file in video_dict['FileNames']:
        
        print ('Processing File: {f}'.format(f=file))  
        video_dict['file'] = file #used both to set the path and to store filenames when saving
        video_dict['fpath'] = os.path.join(os.path.normpath(video_dict['dpath']), file)
        
        reference = Reference(video_dict,crop,num_frames=100) 
        location = TrackLocation(video_dict,tracking_params,reference,crop)
        if region_names!=None:
            location = ROI_Location(reference,poly_stream,region_names,location)
        location.to_csv(os.path.splitext(video_dict['fpath'])[0] + '_LocationOutput.csv')
        file_summary = Summarize_Location(location, video_dict, bin_dict=bin_dict, region_names=region_names)
               
        try: #Add summary info for individual file to larger summary of all files
            summary_all = pd.concat([summary_all,file_summary])
        except NameError: #to be done for first file in list, before summary_all is created
            summary_all = file_summary
        
        #Plot Heat Map
        image = hv.Image((np.arange(reference.shape[1]), np.arange(reference.shape[0]), reference)).opts(
        width=int(reference.shape[1]*stretch['width']),
        height=int(reference.shape[0]*stretch['height']),
        invert_yaxis=True,cmap='gray',toolbar='below',
        title=file+": Motion Trace")
        points = hv.Scatter(np.array([location['X'],location['Y']]).T).opts(color='navy',alpha=.2)
        heatmaps.append(image*poly*points) if poly_stream!=None else heatmaps.append(image*points)

    #Write summary data to csv file
    sum_pathout = os.path.join(os.path.normpath(video_dict['dpath']), 'BatchSummary.csv')
    summary_all.to_csv(sum_pathout)
    
    layout = hv.Layout(heatmaps)
    return layout

########################################################################################        

def PlayVideo(video_dict,display_dict,crop,location):

    #Load Video and Set Saving Parameters
    cap = cv2.VideoCapture(video_dict['fpath'])#set file\
    if display_dict['save_video']==True:
        ret, frame = cap.read() #read frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        try: #if crop is supplied
            Xs=[crop.data['x0'][0],crop.data['x1'][0]]
            Ys=[crop.data['y0'][0],crop.data['y1'][0]]
            fxmin,fxmax=int(min(Xs)), int(max(Xs))
            fymin,fymax=int(min(Ys)), int(max(Ys))
            frame = frame[fymin:fymax,fxmin:fxmax]
        except: #if no cropping is used
            fxmin,fxmax=0,frame.shape[1]
            fymin,fymax=0,frame.shape[0]
        width = int(frame.shape[1])
        height = int(frame.shape[0])
        fourcc = 0#cv2.VideoWriter_fourcc(*'jpeg') #only writes up to 20 fps, though video read can be 30.
        writer = cv2.VideoWriter(os.path.join(os.path.normpath(video_dict['dpath']), 'video_output.avi'), 
                                 fourcc, 20.0, 
                                 (width, height),
                                 isColor=False)

    #Initialize video play options   
    cap.set(1,video_dict['start']+display_dict['start']) #set starting frame
    rate = int(1000/video_dict['fps']) #duration each frame is present for, in milliseconds

    #Play Video
    for f in range(display_dict['start'],display_dict['stop']):
        ret, frame = cap.read() #read frame
        if ret == True:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            try: #if crop is supplied
                Xs=[crop.data['x0'][0],crop.data['x1'][0]]
                Ys=[crop.data['y0'][0],crop.data['y1'][0]]
                fxmin,fxmax=int(min(Xs)), int(max(Xs))
                fymin,fymax=int(min(Ys)), int(max(Ys))
            except: #if no cropping is used
                fxmin,fxmax=0,frame.shape[1]
                fymin,fymax=0,frame.shape[0]
            frame = frame[fymin:fymax,fxmin:fxmax]
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

        
def PlayVideo2(video_dict,display_dict,crop,location):
    cap = cv2.VideoCapture(video_dict['fpath'])
    
    
########################################################################################        
#Code to export svg
#conda install -c conda-forge selenium phantomjs

#import os
#from bokeh import models
#from bokeh.io import export_svgs

#bokeh_obj = hv.renderer('bokeh').get_plot(image).state
#bokeh_obj.output_backend = 'svg'
#export_svgs(bokeh_obj, dpath + '/' + 'Calibration_Frame.svg')

    