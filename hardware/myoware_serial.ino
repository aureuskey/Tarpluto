// Tarpluto: stream MyoWare 2.0 envelope (ENV pin -> A0) over serial at ~100 Hz.
void setup() { Serial.begin(115200); }
void loop() { Serial.println(analogRead(A0)); delay(10); }
