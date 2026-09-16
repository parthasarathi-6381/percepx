/* Temporary diagnostic sketch -- reads the Bosch chip-ID register (0xD0)
 * directly over I2C, bypassing any library's begin()/address assumptions.
 * Known IDs at that register: 0x55 = BMP180, 0x58 = BMP280, 0x60 = BME280.
 * Not part of the main project; flash it, read Serial Monitor at 9600 baud,
 * then reflash env_lidar_monitor.ino when done.
 */
#include <Wire.h>

uint8_t readChipId(uint8_t addr) {
  Wire.beginTransmission(addr);
  Wire.write(0xD0);
  if (Wire.endTransmission(false) != 0) return 0xFF;  // no ACK at this address
  Wire.requestFrom(addr, (uint8_t)1);
  if (Wire.available()) return Wire.read();
  return 0xFF;
}

void printResult(uint8_t addr) {
  uint8_t id = readChipId(addr);
  Serial.print("Addr 0x");
  Serial.print(addr, HEX);
  Serial.print(" -> chip ID 0x");
  Serial.println(id, HEX);
}

void setup() {
  Wire.begin();
  Serial.begin(9600);
  while (!Serial) {}
  Serial.println("Reading Bosch chip-ID register (0xD0)...");
  printResult(0x76);
  printResult(0x77);
  Serial.println("Known IDs: 0x55=BMP180  0x58=BMP280  0x60=BME280  0xFF=no ACK");
}

void loop() {}
