// Hardware Components
#define RELAY_PUMP          18
#define WS2812B_PIN         17
#define LIGHT_SENSOR        16
#define DHT_11_SENSOR       19
#define HUMIDITY_SENSOR_1   34
#define HUMIDITY_SENSOR_2   35

// Blynk variable const
#define BLYNK_TEMPLATE_ID   "TMPL6t1UuDZXD"
#define BLYNK_TEMPLATE_NAME "SmartWatering"
#define BLYNK_AUTH_TOKEN    "ATIa0ZEmi0ouwZwlDqBKWwPI858HoEcv"
#define BLYNK_PRINT         Serial

// WiFi const
#define WIFI_SSID "WiFi_TEST"
#define WIFI_PASS "admin123456"

//W2812B RGB LED
#define NUM_LEDS      24      

int auto_mode = 0;
int bump_relay = 0;
int r = 0, g = 0, b = 0;

const float MAX_TEMP = 35.0;
const float MIN_SOIL_HUMIDITY = 40.0;