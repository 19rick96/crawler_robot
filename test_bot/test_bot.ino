#include <Servo.h>

Servo joint1;
//Servo joint2;

void setup() 
{
  joint1.attach(9);
  //joint2.attach();
  joint1.write(0);
    delay(5);
  Serial.begin(9600);
}

byte a1[2];

void loop() 
{
  if(Serial.available()>0)
  {
    Serial.readBytes(a1,2);
    //Serial.println(a1[0]);
    //Serial.println(a1[1]);
    
    joint1.write(a1[0]);
    delay(4);
    
  }
}
