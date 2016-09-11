void setup() 
{
  pinMode(13,OUTPUT);
  digitalWrite(13,LOW);

  Serial.begin(9600);
}

void loop() 
{
  if(Serial.available() > 0)
  {
    char letter = Serial.read();

    if(letter == '1')
      digitalWrite(13,HIGH);
    if(letter == '0')
      digitalWrite(13,LOW);
  }
}
