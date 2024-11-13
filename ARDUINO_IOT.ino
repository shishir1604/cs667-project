#include "SoftwareSerial.h"
#include "Plantower_PMS7003.h"
#include <DHT.h>

// DHT sensor setup
#define DHTPIN 4  // Pin for DHT11 data
DHT dht(DHTPIN, DHT11);

// PMS7003 and SoftwareSerial setup
SoftwareSerial esp(2, 3);  // RX, TX for communication with ESP
Plantower_PMS7003 pms7003;

int sensorValue;
int humidity = 0;
int temp = 0;
int ammonia = 0;
int benzene = 0;
int pm1_0 = 0;
int pm2_5 = 0;
int pm10 = 0;

void setup() {
  Serial.begin(9600);      // Serial monitor for debugging
  esp.begin(9600);         // Serial communication with ESP
  pms7003.init(&Serial);   // Initialize PMS7003 sensor on default Serial
  dht.begin();             // Initialize DHT11 sensor
}

void loop() {
  // Get data from DHT11, MQ135, and PMS7003 sensors
  readDHT11();
  readMQ135();
  readPMS7003();
  // Send all data to ESP
  sendDataToESP();

  delay(5000);  // Delay for the next reading
}

// Function to read DHT11 values
void readDHT11() {
  humidity = dht.readHumidity();
  temp = dht.readTemperature();
}

// Function to read MQ135 values for benzene and ammonia
void readMQ135() {
  sensorValue = analogRead(0);
  ammonia = sensorValue * 0.03;  // Approximated conversion for ammonia
  benzene = sensorValue * 0.02;  // Approximated conversion for benzene
}

// Function to read PMS7003 data
void readPMS7003() {
  pms7003.updateFrame();
  if (pms7003.hasNewData()) {
    pm1_0 = pms7003.getPM_1_0();
    pm2_5 = pms7003.getPM_2_5();
    pm10 = pms7003.getPM_10_0();
  }
}

// Function to send all collected data to ESP using String
void sendDataToESP() {
  // Create a String object and append all sensor data
  String output = "TEMP:" + String(temp) +
                  ",HUMIDITY:" + String(humidity) +
                  ",PM2.5:" + String(pm2_5) +
                  ",PM10:" + String(pm10) +
                  ",BENZENE:" + String(benzene) +
                  ",AMMONIA:" + String(ammonia) + "\n";

  // Send the formatted data to the ESP via SoftwareSerial
  esp.println(output);
  Serial.println(output);  // Print to Serial Monitor for debugging
}
