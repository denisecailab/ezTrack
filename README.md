# Behavior Tracking with ezTrack
This repository contains iPython files that can be used to track the location, motion, and freezing of an animal. For the sake of clarity, these processes are described as two modules: one for the analysis of freezing; the other for tracking an animal's location.  **If you are unfamiliar with how to use iPython/Jupyter Notebook, please see [Getting Started](https://github.com/DeniseCaiLab/GettingStarted)**.

**Please see our biorxiv preprint (https://www.biorxiv.org/content/10.1101/592592v2)** for ezTrack video tutorials (Supplementary Videos 1 and 2) and for validation of tracking.

# Installation and Package Requirements
The iPython scripts included in this repository require the following packages to be installed in your Conda environment.  Although the package versions used during development are listed it is likely that latest releases of all will be fine to use:
* python (3.6.7)
* jupyter (1.0.0)
* numpy (1.15.2)
* pandas (0.23.0)
* matplotlib (3.1.1) 
* opencv (3.4.3)
* holoviews (1.12.3)
* scipy (1.2.1)

1. [Download and install Miniconda/Conda](https://conda.io/miniconda.html).  Download version with Python 3.7.  The ‘.pkg’ installer is likely easier for those unfamiliar with coding. See below for an explanation of Conda.

2. Create the ezTrack Conda environment:

    a. For OSX/Linux Users, open Terminal on your computer. If using Windows, open Anaconda Prompt.
    
    b. Copy the following command into Terminal/AnacondaPrompt: ` conda create -n ezTrack -c conda-forge python=3.6 pandas=0.23.0 matplotlib=3.1.1 opencv=3.4.3 jupyter=1.0.0 holoviews=1.12.3 scipy=1.2.1` and hit enter.
    
    c. When prompted with “Proceed ([y]/n)? Type in `y` and hit enter.
    
    d. The ezTrack Conda Environment is now installed.

3. From the ezTrack github page, download all files onto your computer. On the main page of the repository, click the button to ‘Clone or download’, and download the zip folder onto your hard drive (don't forget to unzip the folder). The unzipped folder is called ezTrack-master and contains all files necessary to run ezTrack.  Alternatively, use git commands if you are familiar with them (`git clone https://github.com/DeniseCaiLab/ezTrack.git`)

4. Activate the ezTrack Conda environment from Terminal/Anaconda Prompt.  On OSX/Linux, use your finder to open Terminal and enter the command: `source activate ezTrack`.  On a Windows os, search for 'Anaconda Prompt'.  Once open, enter the command `conda activate ezTrack`.  

5. Launch Jupyter Notebook from Terminal/Anaconda Prompt (command: `jupyter notebook`). 

**Note that steps 4-5 must be done fresh each time ezTrack is used on your computer.

6. **From within Jupyter Notebook**, navigate to the ezTrack folder you downloaded and open the desired .ipynb file.  You can now run the code!  If you're new to Jupyter Notebook, you might check out some online tutorials.  There are tons.  [Here's one very simple one.](https://nbviewer.jupyter.org/github/jupyter/notebook/blob/master/docs/source/examples/Notebook/Notebook%20Basics.ipynb)

# Location Tracking Module
The location tracking module allows for the analysis of a single animal's location on a frame by frame basis.  In addition to providing the user the with the ability to crop the portion of the video frame in which the animal will be, it also allows the user to specify regions of interest (e.g. left and right sides) and provides tools to quantify the time spent in each region, as well as distance travelled.  

### Basic Workflow for Location Tracking
1. Process several individual behavior videos with **LocationTracking_Individual.ipynb**.  This will allow extensive visualization of results in order to ensure confidence in selected parameters. 
2. Once you are comfortable with parameters, use **LocationTracking_Batch.ipynb** on a whole folder of videos.

**Note:** LocationTracking_Functions.py must be in the same folder as LocationTracking_Individual.ipynb and LocationTracking_Batch.ipynb in order for them to work.

![Optional Text](../master/Images/LocationTracking_Schematic.png)

# Freeze Analysis Module
The freeze analysis module allows the user to automatically score an animal's motion and freezing while in a conditioning chamber.  It was designed with side-view recording in mind, and with the intention of being able to crop the top of a video frame to remove the influence of fiberoptic/miniscope cables.  In the case where no cables are to be used, recording should be capable from above the animal.  

### Basic Workflow for Freeze Analysis
1. Run **FreezeAnalysis_Calibration.ipynb** on a short video of a chamber with no animal in it (~10 sec).  This allows detection of basal fluctuation in pixel grayscale values.  A suggested cutoff for use with subsequent steps is provided.
2. Process several individual behavior videos with **FreezeAnalysis_Individual.ipynb**.  This will allow extensive visualization of results in order to ensure confidence in selected parameters. 
3. Once you are comfortable with parameters, use **FreezeAnalysis_BatchProcess.ipynb** on a whole folder of videos.

**Note:** FreezeAnalysis_Functions.py must be in the same folder as FreezeAnalysis_individual.ipynb and FreezeAnalysis_BatchProcess.ipynb in order for them to work.

![Optional Text](../master/Images/FreezeAnalysis_Schematic.png)

## Video requirements
As of yet, mpg1, wmv, and avi (mp4 codec) all work.  Many more should work but have not yet been tested.  

## Running Code
After downloading the files onto your local computer in a single folder, from the Terminal/Anaconda Prompt activate the necessary Conda environment (```source activate ezTrack``` // in windows: ```conda activate ezTrack```) and open Jupyter Notebook (```jupyter notebook```), then navigate to the files on your computer. The individual scripts contain more detailed instructions.

## Web Browser Compatibility
We have most extensively tested ezTrack using Chrome, and Firefox to a lesser extent.  Some issues have been found with Internet Explorer and we recommend that it be avoided when usng ezTrack.

## Why Conda? What is a Conda Environment?
Conda is a package and environment manager that provides a convenient way to utilize different versions of software and software packages for different applications, solving compatibility issues.  To give an example, say a particular application requires Python 2.7 and SuperPackage 2.4, while another requires Python 3.7 and SuperPackage 3.8. Using both applications could quickly become inconvenient. Conda allows one to create different ‘environments’ on your computer, each with different versions of Python and any other packages (here, SuperPackage), which you can then flexibly activate to use different applications.  Moreover, because Conda maintains prior versions of packages, this helps ensure that software remains usable even when new package releases could otherwise give rise to compatibility issues.


