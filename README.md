<p align="center">
  <img width="150" src="./Images/KathleenWang_for_ezTrack.png">
</p>

# Behavior Tracking with ezTrack
This page hosts iPython files that can be used to track the location, motion, and freezing of an animal. For the sake of clarity, these processes are described as two modules: one for tracking an animal's location; the other  for the analysis of freezing.  **If you are unfamiliar with how to use iPython/Jupyter Notebook, please see [Getting Started](https://github.com/DeniseCaiLab/GettingStarted)**.

![Examples](../master/Images/Examples.gif)

# Please cite ezTrack if you use it in your research:
Pennington ZT, Dong Z, Feng Y, Vetere LM, Page-Harley L, Shuman T, Cai DJ (2019). ezTrack: An open-source video analysis pipeline for the investigation of animal behavior. Scientific Reports: 9(1): 19979


# Check out the ezTrack wiki
For instructions on installation and use, go [here](https://github.com/denisecailab/ezTrack/wiki).

# New Feature Alerts:
- 04/11/2021: ezTrack now has **algorithm for removing wires** in the location tracking module.
- 07/20/2020: ezTrack now supports **spatial downsampling** of videos!  You can reduce the resolution of the video to greatly speed processing. Processing high-definition videos on older laptops/desktops can be slow, but by downsampling, processing speeds are much faster.
- 07/19/2020: Location tracking module now allows user to **manually define frame numbers to be used when selecting reference**.  This is useful if baseline portion of video without animal will be used for reference, and resolves issue when alternative video being used for reference is a different length than the video being processed.
- 06/16/2020:  Location tracking module **now allows user to define regions of frame that they would like excluded from the analysis**.  This is useful in situations where an extraneous object enters into periphery, or even center, of the field of view.


# Location Tracking Module
The location tracking module allows for the analysis of a single animal's location on a frame by frame basis.  In addition to providing the user the with the ability to crop the portion of the video frame in which the animal will be, it also allows the user to specify regions of interest (e.g. left and right sides) and provides tools to quantify the time spent in each region, as well as distance travelled.  
![schematic_lt](../master/Images/LocationTracking_Schematic.png)


# Freeze Analysis Module
The freeze analysis module allows the user to automatically score an animal's motion and freezing while in a conditioning chamber.  It was designed with side-view recording in mind, and with the intention of being able to crop the top of a video frame to remove the influence of fiberoptic/miniscope cables.  In the case where no cables are to be used, recording should be capable from above the animal.  
![schematic_fz](../master/Images/FreezeAnalysis_Schematic.png)


# License
This project is licensed under GNU GPLv3.
