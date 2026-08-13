#include "src/GenericSerial/GenericSerial.h"

#define BAUDRATE 115200


// Adafruit_MPR121 cap = Adafruit_MPR121();
GenericSerial gs = GenericSerial();
uint16_t lasttouched = 0;
uint16_t currtouched = 0;


//
//COMMENT THIS SECTION OUT IF NOT USING DIGITAL INPUTS ON ARDUINO
//
int INPUTS[] = {2, 3}; // define pins of digital inputs used on arduino. note that this should not overlap with capacitive touch sensor port numbers
int lastinput [sizeof(INPUTS) / sizeof(int)] = {1}; //prior input states
int currinput [sizeof(INPUTS) / sizeof(int)] = {1}; //current input states
int input_num = sizeof(INPUTS) / sizeof(int);


void setup()
{

  gs.begin(BAUDRATE);

  //
  //COMMENT THIS SECTION OUT IF NOT USING CAPACITIVE TOUCH SENSOR
  //
  // while (!cap.begin(0x5A))
  // {
  //   ;
  // }
  // cap.setThreshholds(12, 6);

}

void loop()
{
  
  gs.process();

  //
  //COMMENT THIS SECTION OUT IF NOT USING CAPACITIVE TOUCH SENSOR
  //
  // currtouched = cap.touched();
  // for (uint8_t i = 0; i < 12; i++)
  // {
  //   // it if *is* touched and *wasnt* touched before, alert!
  //   if ((currtouched & _BV(i)) && !(lasttouched & _BV(i)))
  //   {
  //     byte buf[2] = {i, 1};
  //     gs.send(buf);
  //   }
  //   // if it *was* touched and now *isnt*, alert!
  //   if (!(currtouched & _BV(i)) && (lasttouched & _BV(i)))
  //   {
  //     byte buf[2] = {i, 0};
  //     gs.send(buf);
  //   }
  // }
  // lasttouched = currtouched;


  //
  //COMMENT THIS SECTION OUT IF NOT USING DIGITAL INPUTS ON ARDUINO
  //
  for (int i = 0; i < input_num; i++) {
    currinput[i] = digitalRead(INPUTS[i]); 
    if ((currinput[i]==1) && (currinput[i] != lastinput[i])) { 
      //digitalWrite(13, LOW); //this can be used for testing output state with LED
      Serial.write(INPUTS[i]);
      Serial.write(currinput[i]);
      Serial.write(255);
    }
    if ((currinput[i]==0) && (currinput[i] != lastinput[i])) {
      //digitalWrite(13, HIGH); //this can be used for testing output state with LED
      Serial.write(INPUTS[i]);
      Serial.write(currinput[i]);
      Serial.write(255);
    }
    lastinput[i] = currinput[i];
  }
  
  
}
