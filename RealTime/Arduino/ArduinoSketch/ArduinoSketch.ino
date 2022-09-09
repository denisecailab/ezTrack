#include "src/GenericSerial/GenericSerial.h"

#ifndef _BV
#define _BV(bit) (1 << (bit))
#endif

#define BAUDRATE 115200

GenericSerial gs = GenericSerial();

void setup()
{
  gs.begin(BAUDRATE);
}

void loop()
{
  gs.process();
}
