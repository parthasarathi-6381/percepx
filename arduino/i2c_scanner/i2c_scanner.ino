/* Temporary diagnostic sketch -- scans the I2C bus and prints every address
 * that responds. Used to check whether the BMP280 (expected at 0x76 or
 * 0x77) is actually alive on the bus, independent of what the
 * Adafruit_BMP280 library's begin() thinks. Not part of the main project;
 * flash it, read the Serial Monitor at 9600 baud, then reflash
 * env_lidar_monitor.ino when done.
 */
#include <Wire.h>

void setup() {
  Wire.begin();
  Serial.begin(9600);
  while (!Serial) {}
  Serial.println("I2C scan starting...");
}

void loop() {
  byte count = 0;
  for (byte addr = 1; addr < 127; addr++) {
    Wire.beginTransmission(addr);
    byte err = Wire.endTransmission();
    if (err == 0) {
      Serial.print("Found device at 0x");
      if (addr < 16) Serial.print("0");
      Serial.println(addr, HEX);
      count++;
    }
  }
  if (count == 0) Serial.println("No I2C devices found.");
  Serial.println("---");
  delay(3000);
}
