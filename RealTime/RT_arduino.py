
"""

LIST OF CLASSES/FUNCTIONS

Arduino (Class)

"""


import os
import sys
import time
import serial
import threading
import queue
import datetime
import time
import warnings
from threading import Thread



class Arduino():
    
    """ 
    -------------------------------------------------------------------------------------
    
    Base class for working with Arduino in order to transmit and receive digital signals.

    -------------------------------------------------------------------------------------
    
    Methods:
    
        - init
        - initialize
        - stop
        - io_config
        - io_transmitter
        - io_send
        - digitalHigh
        - digitalLow
        - handshake
    
    Attributes (see __init__ for details):
    
        - ser
        - port
        - keys_dout
        - cmnds
        - cmndflg
        - q_tasks
        - state
 
    """
    
    def __init__(self, port, keys_dout=None, baudrate=115200, timeout=1):
        
        """ 
        -------------------------------------------------------------------------------------

        Arduino class initialization

        -------------------------------------------------------------------------------------
        Args:
            port:: [str]
                Arduino port address.  Can be found if Arduino IDE if unfamiliar with Terminal
                commands for finding this.

            keys_dout:: [dict]
                Should be dictionary where each key is the name for a digital output and each
                key is the pin ID.

            baudrate:: [unsigned integer]
                Baudrate for communicating with Arduino.
                
            timeout:: [unsigned integer]
                Millisecond timeout for Arduino
                
        -------------------------------------------------------------------------------------        
        Attributes:
            ser:: [serial.Serial]
                pySerial.Serial connection for Arduino communication.
                
            keys_dout::
                Dictionary where each key is the name for a digital output and each
                key is the pin ID.
            
            cmnds:: [dict]
                Dictionary containing values for transmitting signals to Arduino. Note that the 
                meaning of these signals (currently 0,1,2,255) is set on the Arduino side.
            
            cmndflg:: [bytes]
                Flag indicating end of command signal in pySerial buffer. Do not change unless
                also changing Arduino/GenericSerial files.
            
            q_tasks:: [queue.Queue]
                Thread queue for holding commands until they are to be sent.
            
            state::
        
        -------------------------------------------------------------------------------------        
        Notes:
        
        """

        self.ser = serial.Serial(port = port, baudrate = baudrate, timeout = timeout)
        self.keys_dout = keys_dout
        self.cmnds = dict(low=0, high=1, setout=2)
        self.cmndflg = bytes([255])
        self.q_tasks = queue.Queue()
        self.state = 'uninitiated'

        
        
    def initialize(self):
        
        """ 
        -------------------------------------------------------------------------------------
        
        Initialized connection with Arduino. Tests communication, configures inputs/outputs,
        and starts thread for transmitting commands to Arduino
        
        -------------------------------------------------------------------------------------
        Notes:

        """
        
        self.handshake()
        self.io_config()
        Thread(target=self.io_transmitter, args=()).start()
        print('state: ready')
        self.state = 'ready'
       
    
    
    def stop(self):
        
        """ 
        -------------------------------------------------------------------------------------
        
        Closes serial port and stops thread for transmitting commands to Arduino
        
        -------------------------------------------------------------------------------------
        Notes:

        """
        
        if self.state=='ready':
            self.state = 'stopped'
        else:
            self.ser.close()
    
    
    
    def io_config(self):
        
        """ 
        -------------------------------------------------------------------------------------
        
        Configures Arduino digital inputs and outputs
        
        -------------------------------------------------------------------------------------
        Notes:
        digital inputs have yet to be implemented.

        """
        
        if self.keys_dout is not None:
            for name, pin in self.keys_dout.items():
                ts = time.time()
                self.io_send((pin, self.cmnds['setout']), ts)
            print('outputs configured')
      
    
    
    def io_transmitter(self, timeout=.001):
        
        """ 
        -------------------------------------------------------------------------------------
        
        Controls transmission of signals to arduino
        
        -------------------------------------------------------------------------------------
        Notes:

        """
        
        while True:
            cur_ts = time.time()
            hold_tasks = []
            while not self.q_tasks.empty():
                ts, cmd = self.q_tasks.get()
                if ts <= cur_ts:
                    cmd = cmd + self.cmndflg
                    self.ser.write(cmd)
                else:
                    hold_tasks.append((ts, cmd))
            for tsk in hold_tasks:
                self.q_tasks.put(tsk)
            if self.state=='stopped':
                for name, pin in self.keys_dout.items():
                    self.digitalLow(pin)
                self.ser.close()
                break
            time.sleep(timeout)
    
    
    
    def io_send(self, sig, ts):
        
        """ 
        -------------------------------------------------------------------------------------
        
        Adds tasks to task queue, to be executed by io_transmitter thread.
        
        -------------------------------------------------------------------------------------
        Args:
            sig:: [tuple]
                Tuple of length 2 where first index should be 
                
            ts:: [float]
                Timestamp of signal
        
        -------------------------------------------------------------------------------------
        Notes:

        """
        
        self.q_tasks.put((ts, bytes(sig)))
    
    
    
    def digitalHigh(self, pin, hold=None):
        
        """ 
        -------------------------------------------------------------------------------------
        
        Sets digital output pins to high. Can be done for set period of time. 
        
        -------------------------------------------------------------------------------------
        Args:
            pin:: [int or string]
                Either an integer, specifying output pin, or a string that serving as key in
                Arduino.keys_dout
                
            hold:: [float]
                Duration, in seconds, before signal is reversed.
        
        -------------------------------------------------------------------------------------
        Notes:

        """
        
        cur_ts = time.time()
        if type(pin) is str:
            pin = self.keys_dout[pin]
        self.io_send((pin, self.cmnds['high']), cur_ts)
        if hold is not None:
            self.io_send((pin, self.cmnds['low']), cur_ts + hold)
    
    
    
    def digitalLow(self, pin, hold=None):
        
        
        """ 
        -------------------------------------------------------------------------------------
        
        Sets digital output pins to low. Can be done for set period of time. 
        
        -------------------------------------------------------------------------------------
        Args:
            pin:: [int or string]
                Either an integer, specifying output pin, or a string that serving as key in
                Arduino.keys_dout
                
            hold:: [float]
                Duration, in seconds, before signal is reversed.
        
        -------------------------------------------------------------------------------------
        Notes:

        """
        
        cur_ts = time.time()
        if type(pin) is str:
            pin = self.keys_dout[pin]
        self.io_send((pin, self.cmnds['low']), cur_ts)
        if hold is not None:
            self.io_send((pin, self.cmnds['high']), cur_ts + hold)

            
            
    def handshake(self, timeout=5):
        
        """ 
        -------------------------------------------------------------------------------------
        
        Tests for bi-directional communication with Arduino.
        
        -------------------------------------------------------------------------------------
        Args:
            timeout:: [float]
                Duration, in seconds, to wait for signal from Arduino.

        -------------------------------------------------------------------------------------
        Notes:

        """
        
        ts = time.time()
        while True:
            self.ser.write(self.cmndflg)
            data = self.ser.read()
            if data == self.cmndflg:
                print('handshake success')
                break
            if time.time()-ts > timeout:
                print('timeout without connection')
                break



