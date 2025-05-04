// define the pins
int PIR = 7;
int Buzzer = 2;

void setup() {
  pinMode(Buzzer, OUTPUT);
  pinMode(PIR, INPUT);

  Serial.begin(9600);
  while (!Serial);  // Wait for serial to be ready (esp. for Leonardo/Micro boards)
}

void loop() {
  // --- PIR Sensor + Buzzer Control ---
  int value = digitalRead(PIR);
  if (value == HIGH) {
    digitalWrite(Buzzer, HIGH);
  } else {
    digitalWrite(Buzzer, LOW);
  }

  // --- Serial Communication ---
  if (Serial.available() > 0) {
    char received = Serial.read();
    
    if (received == '1') {
      Serial.println("Hi Raspberry Pi!");
    }
    // you can add more commands here if you want later
  }
}
