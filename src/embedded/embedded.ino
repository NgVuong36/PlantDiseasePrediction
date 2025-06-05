#include <DHT.h>
#include "config.h"
#include <WiFi.h>
#include <WiFiClient.h>
#include <BlynkSimpleEsp32.h>
#include <Adafruit_NeoPixel.h>

BlynkTimer timer;
DHT dht(DHT_11_SENSOR, DHT11);
Adafruit_NeoPixel strip(NUM_LEDS, WS2812B_PIN, NEO_GRB + NEO_KHZ800);

bool lastPumpState = false;

void updateRGB() {
  for (int i = 0; i < strip.numPixels(); i++) strip.setPixelColor(i, strip.Color(r, g, b));
  Serial.printf("RGB Updated: (%d, %d, %d)\n", r, g, b);
  strip.show();
}

BLYNK_WRITE(V4) {
  bump_relay = param.asInt();
  if (!auto_mode) digitalWrite(RELAY_PUMP, bump_relay);
}

BLYNK_WRITE(V5) {
  r = param.asInt();
  updateRGB();
}

BLYNK_WRITE(V6) {
  g = param.asInt();
  updateRGB();
}

BLYNK_WRITE(V7) {
  b = param.asInt();
  updateRGB();
}

BLYNK_WRITE(V9) {
  auto_mode = param.asInt();
}

void sendData() {
  float air_humidity = dht.readHumidity();
  float room_temperature = dht.readTemperature();
  float humidity_1 = map(analogRead(HUMIDITY_SENSOR_1), 0, 4095, 0, 100);
  float humidity_2 = map(analogRead(HUMIDITY_SENSOR_2), 0, 4095, 0, 100);

  if (isnan(air_humidity) || isnan(room_temperature)) return;

  Blynk.virtualWrite(V0, humidity_1);
  Blynk.virtualWrite(V1, humidity_2);
  Blynk.virtualWrite(V2, air_humidity);
  Blynk.virtualWrite(V3, room_temperature);

  Serial.printf("Soil1: %.1f%%, Soil2: %.1f%%, AirHum: %.1f%%, Temp: %.1f°C\n",
                humidity_1, humidity_2, air_humidity, room_temperature);

  if (auto_mode) {
    bool shouldWater = (humidity_1 < MIN_SOIL_HUMIDITY || 
                       humidity_2 < MIN_SOIL_HUMIDITY || 
                       room_temperature > MAX_TEMP);
    
    if (shouldWater != lastPumpState) {
      if (shouldWater) {
        digitalWrite(RELAY_PUMP, HIGH);
        Blynk.logEvent("turn_bump_on", "Bat may bom, vi do am thap");
      } else {
        digitalWrite(RELAY_PUMP, LOW);
        Blynk.logEvent("turn_bump_on", "Da du do am, tat may bom!");
      }
      lastPumpState = shouldWater;
    }
    
    Blynk.virtualWrite(V4, shouldWater ? 1 : 0);
  }
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  pinMode(RELAY_PUMP, OUTPUT);
  pinMode(LIGHT_SENSOR, INPUT);
  pinMode(HUMIDITY_SENSOR_1, INPUT);
  pinMode(HUMIDITY_SENSOR_2, INPUT);
  
  digitalWrite(RELAY_PUMP, LOW);

  strip.begin();
  strip.setBrightness(100);
  updateRGB();

  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Blynk.begin(BLYNK_AUTH_TOKEN, WIFI_SSID, WIFI_PASS);

  dht.begin();
  timer.setInterval(2000L, sendData);
}

void loop() {
  Blynk.run();
  timer.run();
}