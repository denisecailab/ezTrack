/*
  genericSerial.cpp - Library for simple digital control over serial.
  Created by Phil Dong, July 11, 2021.
  Commented by Zach Pennington, Sep 6, 2022
*/
#include "Arduino.h"
#include "GenericSerial.h"

GenericSerial::GenericSerial() {}

void GenericSerial::begin(long baud)
/*
  The following initializes serial connection and assures pyserial connection
  After awaiting pyserial command signal, sends command signal back.
*/
{
    Serial.begin(baud);
    while (true)
    {
        if (Serial.available())
        {
            int data;
            data = Serial.read();
            if (data == CMD_FLAG)
            {
                Serial.write(CMD_FLAG);
                break;
            }
        }
    }
}

void GenericSerial::process()
/*
  Continuous loop that reads commands in buffer and executes them.
*/
{
    if (Serial.available() > 2)
    {
        size_t sigbyte = Serial.readBytesUntil(CMD_FLAG, this->_buffer, 3);
        if (sigbyte == 3)
        {
            switch (this->_buffer[1])
            {
            case CMD_MODE_OUT:
                pinMode(this->_buffer[0], OUTPUT);
                break;
            case CMD_WRITE_LOW:
                digitalWrite(this->_buffer[0], LOW);
                break;
            case CMD_WRITE_HIGH:
                digitalWrite(this->_buffer[0], HIGH);
                break;
            default:
                return;
            }
        }
    }
}

void GenericSerial::send(byte buf[])
/*
  Command for Arduino to write to buffer.
  Currently not used.
*/
{
    size_t len = sizeof(buf);
    Serial.write(buf, len);
    Serial.write(CMD_FLAG);
}