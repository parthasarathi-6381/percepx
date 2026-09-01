/* ==========================================================================
 * Grove Beginner Kit for Arduino -- Environmental Monitor for LiDAR
 * data-quality risk (rain / fog / smoke / dust proxy sensing).
 *
 * BOARD: Seeeduino (Grove Beginner Kit for Arduino), ATmega328P core.
 *
 * WIRING (Grove Beginner Kit on-board sensors + one added sensor):
 *   Sound Sensor              -> A2   (moved here from its on-board default
 *                                      of A0, swapped with the potentiometer)
 *   Air Quality Sensor v1.3   -> A0   (ADDED -- wired into the Sound
 *                                      Sensor's original socket; the
 *                                      on-board potentiometer was removed
 *                                      entirely to free up this rewiring,
 *                                      so there is no OLED page control --
 *                                      the OLED just shows every sensor
 *                                      at once, see render_oled() below)
 *   Temp & Humidity (AHT20/DHT20) -> I2C  (on-board, addr 0x38)
 *   OLED Display 0.96"        -> I2C  (on-board)
 *   Air Pressure (BMP280)     -> I2C  (on-board; tries 0x77 first, falls
 *                                      back to 0x76 -- the address varies
 *                                      by board revision, confirm with an
 *                                      I2C scanner if it still isn't found)
 *   3-Axis Accel (LIS3DHTR)   -> I2C  (on-board, addr 0x19)
 *   Relay (D7), Buzzer (D5), Button (D6), LED (D9) -> on-board, unused here.
 *
 * REQUIRED LIBRARIES (Arduino IDE -> Library Manager):
 *   - "DHT20" (I2C AHT20/DHT20 temp+humidity) -- API used here:
 *     begin() / read() / getTemperature() / getHumidity()
 *   - "Seeed_Arduino_LIS3DHTR" by Seeed Studio
 *   - "Adafruit BMP280 Library" by Adafruit (+ "Adafruit Unified Sensor")
 *   - "U8g2" by oliver (u8g2/u8x8) -- OLED text driver
 *
 * OUTPUT: one JSON line per SEND_INTERVAL_MS over USB Serial at 9600 baud,
 * consumed by src/env_sensors/serial_bridge.py on the PC. Raw values only --
 * all good/moderate/bad classification happens on the PC (configs/config.yaml)
 * so thresholds can be tuned without reflashing. The on-device tags shown on
 * the OLED use their OWN copy of the same limits (see THRESHOLDS below) purely
 * so the tag is visible with no laptop attached -- keep the two in sync.
 * ========================================================================== */

#include <Wire.h>
#include <DHT20.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BMP280.h>
#include <LIS3DHTR.h>
#include <U8x8lib.h>

// ---------------------------------------------------------------------------
// Pins
// ---------------------------------------------------------------------------
#define PIN_SOUND       A2
#define PIN_AIR_QUALITY A0  // external sensor; occupies the Sound Sensor's
                             // original socket (potentiometer removed to
                             // free up A2 for the Sound Sensor's new home)

// ---------------------------------------------------------------------------
// Sensor objects
// ---------------------------------------------------------------------------
DHT20 dht;                        // I2C, addr 0x38
Adafruit_BMP280 bmp;              // I2C, tries 0x77 then 0x76 (see setup())
LIS3DHTR<TwoWire> lis;            // I2C, addr 0x19
U8X8_SSD1306_128X64_NONAME_HW_I2C oled(U8X8_PIN_NONE);

bool bmp_ok = false;
bool lis_ok = false;

// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------
const unsigned long SEND_INTERVAL_MS = 50;    // JSON line rate to PC
const unsigned long OLED_INTERVAL_MS = 100;   // OLED refresh rate
const unsigned long DHT_INTERVAL_MS  = 2000;  // AHT20/DHT20 max ~1 Hz, read every 2s

unsigned long last_send = 0;
unsigned long last_dht  = 0;
unsigned long last_oled = 0;

float cached_temp_c  = NAN;
float cached_hum_pct = NAN;

// Sentinel written into a JSON field when a sensor failed to init/read.
const int SENSOR_FAIL = -999;

// ---------------------------------------------------------------------------
// On-device good/moderate/bad thresholds -- MUST match
// configs/config.yaml -> environmental_sensors.thresholds
// (duplicated here only so the OLED can show a tag with no PC attached)
// ---------------------------------------------------------------------------
struct Limits { float good; float moderate; bool high_is_bad; };

const Limits LIM_SOUND       = {300,  600,  true};   // raw 0-1023, louder = worse
const Limits LIM_HUMIDITY    = {60,   85,   true};   // %RH, higher = fog/rain risk
const Limits LIM_VIBRATION   = {0.05, 0.15, true};   // g, deviation from 1g
const Limits LIM_AIR_QUALITY = {700,  400,  false};  // raw 0-1023, LOWER = worse
                                                      // (Grove AQ v1.3: pollution
                                                      //  drives the signal down)

// 0 = good, 1 = moderate, 2 = bad -- lets the OLED pick both a short tag
// per row and an overall worst-of-all-sensors verdict from the same scale.
uint8_t severity(float value, const Limits& lim) {
  if (lim.high_is_bad) {
    if (value <= lim.good) return 0;
    if (value <= lim.moderate) return 1;
    return 2;
  } else {
    if (value >= lim.good) return 0;
    if (value >= lim.moderate) return 1;
    return 2;
  }
}

const char* TAG_SHORT[] = {"GOOD", "MOD", "BAD"};   // fits the 5-col tag field
const char* TAG_FULL[]  = {"GOOD", "MODERATE", "BAD"};

// Writes a JSON number literal for `value` into buf, or the bare token
// `null` when the reading is missing/failed (NaN) -- keeps the JSON valid
// without fabricating a numeric value for a sensor that isn't there.
void format_or_null(float value, uint8_t decimals, char* buf, size_t buf_size) {
  if (isnan(value)) {
    snprintf(buf, buf_size, "null");
  } else {
    dtostrf(value, 1, decimals, buf);
  }
}

// ---------------------------------------------------------------------------
// setup
// ---------------------------------------------------------------------------
void setup() {
  Serial.begin(9600);
  Wire.begin();

  dht.begin();

  // This board's BMP280 answers at 0x77; fall back to the more common 0x76
  // in case a different board revision is in use.
  bmp_ok = bmp.begin(0x77);
  if (!bmp_ok) {
    bmp_ok = bmp.begin(0x76);
  }
  if (!bmp_ok) Serial.println(F("{\"error\":\"BMP280 not found on I2C bus\"}"));

  lis.begin(Wire, 0x19);
  delay(100);
  lis_ok = lis.isConnection();
  if (lis_ok) {
    lis.setOutputDataRate(LIS3DHTR_DATARATE_25HZ);
    lis.setFullScaleRange(LIS3DHTR_RANGE_2G);
  } else {
    Serial.println(F("{\"error\":\"LIS3DHTR not found\"}"));
  }

  oled.begin();
  oled.setFont(u8x8_font_chroma48medium8_r);
  oled.clear();
  oled.drawString(0, 0, "Env-LiDAR Monitor");
  oled.drawString(0, 2, "Booting sensors...");
  delay(800);
}

// ---------------------------------------------------------------------------
// OLED rendering -- all sensors shown at once, one per row (0.96" OLED is
// 128x64px = 16 cols x 8 rows in this 8x8 text mode, so all 6 readings plus
// an overall verdict fit on screen with no paging needed).
// Row layout: 0 overall risk, 1 blank, 2-5 classified proxies, 6-7 context.
// Columns:    0-3 label | 4-9 value (right-justified) | 11-15 tag
// ---------------------------------------------------------------------------
void render_row(uint8_t row, const char* label, float value, uint8_t decimals,
                 const char* tag) {
  oled.clearLine(row);
  oled.drawString(0, row, label);
  if (isnan(value) || value <= SENSOR_FAIL + 1) {
    oled.drawString(4, row, "N/A");
  } else {
    char valstr[10];
    dtostrf(value, 6, decimals, valstr);
    oled.drawString(4, row, valstr);
  }
  if (tag != nullptr) oled.drawString(11, row, tag);
}

void render_oled(float sound, float temp_c, float hum_pct, float pressure_hpa,
                  float vibration_g, float aq_raw) {
  // Overall verdict = worst of the four proxies with real thresholds
  // (matches the dashboard's "worst reading across all sensors" banner).
  uint8_t worst = 0;
  bool any_reading = false;
  float proxy_values[] = {sound, hum_pct, vibration_g, aq_raw};
  const Limits* proxy_limits[] = {&LIM_SOUND, &LIM_HUMIDITY, &LIM_VIBRATION, &LIM_AIR_QUALITY};

  for (uint8_t i = 0; i < 4; i++) {
    float v = proxy_values[i];
    if (isnan(v) || v <= SENSOR_FAIL + 1) continue;
    any_reading = true;
    uint8_t s = severity(v, *proxy_limits[i]);
    if (s > worst) worst = s;
  }

  char risk_line[17];
  snprintf(risk_line, sizeof(risk_line), "RISK: %s", any_reading ? TAG_FULL[worst] : "--");
  oled.clearLine(0);
  oled.drawString(0, 0, risk_line);
  oled.clearLine(1);

  bool sound_ok = !(isnan(sound) || sound <= SENSOR_FAIL + 1);
  bool hum_ok   = !(isnan(hum_pct) || hum_pct <= SENSOR_FAIL + 1);
  bool vib_ok   = !(isnan(vibration_g) || vibration_g <= SENSOR_FAIL + 1);
  bool aq_ok    = !(isnan(aq_raw) || aq_raw <= SENSOR_FAIL + 1);

  render_row(2, "SND", sound,        0, sound_ok ? TAG_SHORT[severity(sound, LIM_SOUND)] : "--");
  render_row(3, "HUM", hum_pct,      1, hum_ok   ? TAG_SHORT[severity(hum_pct, LIM_HUMIDITY)] : "--");
  render_row(4, "VIB", vibration_g,  3, vib_ok   ? TAG_SHORT[severity(vibration_g, LIM_VIBRATION)] : "--");
  render_row(5, "AQ",  aq_raw,       0, aq_ok    ? TAG_SHORT[severity(aq_raw, LIM_AIR_QUALITY)] : "--");
  render_row(6, "TMP", temp_c,       1, nullptr);
  render_row(7, "PRS", pressure_hpa, 1, nullptr);
}

// ---------------------------------------------------------------------------
// loop
// ---------------------------------------------------------------------------
void loop() {
  unsigned long now = millis();

  // -- AHT20/DHT20 (rate-limited) --
  if (now - last_dht >= DHT_INTERVAL_MS) {
    last_dht = now;
    dht.read();
    float h = dht.getHumidity();
    float t = dht.getTemperature();
    if (h != -999) cached_hum_pct = h;
    if (t != -999) cached_temp_c = t;
  }

  // -- fast sensors, read every loop --
  int sound_raw = analogRead(PIN_SOUND);
  int aq_raw    = analogRead(PIN_AIR_QUALITY);

  float pressure_hpa = bmp_ok ? bmp.readPressure() / 100.0F : SENSOR_FAIL;

  float ax = SENSOR_FAIL, ay = SENSOR_FAIL, az = SENSOR_FAIL, vibration_g = SENSOR_FAIL;
  if (lis_ok) {
    ax = lis.getAccelerationX();
    ay = lis.getAccelerationY();
    az = lis.getAccelerationZ();
    vibration_g = fabs(sqrt(ax * ax + ay * ay + az * az) - 1.0);
  }

  // -- OLED refresh (all sensors shown at once) --
  if (now - last_oled >= OLED_INTERVAL_MS) {
    last_oled = now;
    render_oled(sound_raw, cached_temp_c, cached_hum_pct,
                pressure_hpa, vibration_g, aq_raw);
  }

  // -- JSON line to PC --
  if (now - last_send >= SEND_INTERVAL_MS) {
    last_send = now;

    char s_temp[8], s_hum[8], s_pressure[10], s_ax[8], s_ay[8], s_az[8];
    format_or_null(cached_temp_c, 1, s_temp, sizeof(s_temp));
    format_or_null(cached_hum_pct, 1, s_hum, sizeof(s_hum));
    format_or_null(bmp_ok ? pressure_hpa : NAN, 2, s_pressure, sizeof(s_pressure));
    format_or_null(lis_ok ? ax : NAN, 3, s_ax, sizeof(s_ax));
    format_or_null(lis_ok ? ay : NAN, 3, s_ay, sizeof(s_ay));
    format_or_null(lis_ok ? az : NAN, 3, s_az, sizeof(s_az));

    char buf[200];
    snprintf(buf, sizeof(buf),
      "{\"t\":%lu,\"sound\":%d,\"temp_c\":%s,\"hum_pct\":%s,"
      "\"pressure_hpa\":%s,\"ax\":%s,\"ay\":%s,\"az\":%s,\"aq_raw\":%d}",
      now, sound_raw, s_temp, s_hum, s_pressure, s_ax, s_ay, s_az, aq_raw);
    Serial.println(buf);
  }
}
