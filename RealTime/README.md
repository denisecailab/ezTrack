# RealTime ezTrack

RealTime ezTrack allows for real-time tracking of animals being recorded with standard webcams, and additionally provides options for controlling digital inputs/outputs via an Arduino. Notably, only location tracking has thus far been implemented, and only digital outputs are currently supported.  All code is fully documented, but detailed guides for use have not been implemented.  Minimal examples are provided under `RT_NotebookExample.ipynb` and `RT_ArduinoExample.ipynb`

### Installation
```
conda create -y -n ezTrack -c conda-forge python=3.8 pandas=1.3.2 matplotlib=3.1.1 numpy=1.22.3 opencv=4.5.3 jupyter=1.0.0 holoviews=1.14.5 scipy=1.7.1 scikit-learn=0.24.2 bokeh=2.3.3 jinja2=3.0.3 pyserial=3.5 tqdm
```
