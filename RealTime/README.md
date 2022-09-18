# RealTime ezTrack

RealTime ezTrack allows for real-time tracking of animals being recorded with standard webcams, and additionally provides options for controlling digital inputs/outputs via an Arduino. Notably, only location tracking has thus far been implemented (not freezing), and only digital outputs are currently supported.  All code is fully documented, but detailed guides for use are not complete.  Minimal examples are provided under `RT_NotebookExample.ipynb`, `RT_ArduinoExample.ipynb`, and `EZwithArduino_Example.ipynb`.

### Installation

#### Download/clone ezTrack 

From the ezTrack github page, download all files onto your computer. On the main page of the repository, click the button to ‘Clone or download’, and download the zip folder onto your hard drive (don't forget to unzip the folder). The unzipped folder is called ezTrack-realtime_dev and contains all files necessary to run ezTrack. Alternatively, use git commands if you are familiar with them (git clone https://github.com/DeniseCaiLab/ezTrack.git)

#### Create the ezTrack Conda environment:

a. For OSX/Linux Users, open Terminal on your computer. If using Windows, open Anaconda Prompt.

b. Copy the following command into Terminal/AnacondaPrompt, and then hit enter:
```
conda create -y -n ezTrackRT -c conda-forge python=3.8 pandas=1.3.2 matplotlib=3.1.1 numpy=1.22.3 opencv=4.5.3 jupyter=1.0.0 holoviews=1.14.5 scipy=1.7.1 scikit-learn=0.24.2 bokeh=2.3.3 jinja2=3.0.3 pyserial=3.5 tqdm
```

c. The ezTrack Conda Environment is now installed.

#### Arduino setup instructions

RealTime ezTrack is compatible with multiple Arduino's.  Thus far it has been tested with the Arduino Uno R3, but should work with any Arduino.  If all you need to do is control digital outputs (ttl, etc), the following steps should get your board set up.

a. Download Arduino IDE

b. Upload `ArduinoSketch.ino` onto your Arduino

    b.1. Open Arduino IDE software and connect Arduino to computer
    
    b.2. Go to File -> Open and navigate to ezTrack/RealTime/Arduino/ArduinoSketch/ArduinoSketch.ino
    
    b.3. Make sure your Arduino is the selected board and the port is also correct (Arduino IDE typically does this automatically).
    
    b.4. Click the upload button to upload `ArduinoSketch.ino` to your Arduino.


