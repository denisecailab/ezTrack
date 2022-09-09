/*
  genericSerial.h - Library for simple digital control over serial.
  Created by Phil Dong, July 11, 2021.
*/
#ifndef GENERICSERIAL_H
#define GENERICSERIAL_H

#include "Arduino.h"

#define CMD_FLAG 255     // flag byte to delimit commands and signal success
#define CMD_WRITE_LOW 0  // command to digitalWrite low
#define CMD_WRITE_HIGH 1 // command to digitalWrite high
#define CMD_MODE_OUT 2   // command to set pinMode to output

class GenericSerial
{
public:
  GenericSerial();
  void begin(long baud);
  void process();
  void send(byte buf[]);

private:
  byte _buffer[3];
};

#endif