//define the pins
int PIR = 7;
int Buzzer = 2;

void setup() {
  pinMode(Buzzer, OUTPUT);
  pinMode(PIR, INPUT);
}

void loop() {
  int value = digitalRead(PIR);
  if (value == HIGH) {
    digitalWrite(Buzzer, HIGH);
  }
  else {
    digitalWrite(Buzzer, LOW);
  }
}
