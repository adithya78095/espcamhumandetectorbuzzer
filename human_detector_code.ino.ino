#include "esp_camera.h"
#include <TensorFlowLite_ESP32.h>
#include <TensorFlowLite.h>

#define CAMERA_MODEL_AI_THINKER // Adjust as per your module
#include "camera_pins.h"
#include "model_data.h"  // This is your TensorFlow Lite model data array

#define BUZZER_PIN 13  // GPIO pin for the buzzer

// TensorFlow Lite setup
tflite::MicroErrorReporter tflErrorReporter;
tflite::AllOpsResolver tflOpsResolver;
tflite::MicroInterpreter *tflInterpreter;
TfLiteTensor *inputTensor;
TfLiteTensor *outputTensor;

// Create a tensor arena of 8 KB
constexpr int tensorArenaSize = 8 * 1024;
byte tensorArena[tensorArenaSize];

void setup() {
  Serial.begin(115200);

  pinMode(BUZZER_PIN, OUTPUT);

  // Initialize the model
  const tflite::Model* model = tflite::GetModel(model_data);
  tflInterpreter = new tflite::MicroInterpreter(
    model, tflOpsResolver, tensorArena, tensorArenaSize, &tflErrorReporter);

  tflInterpreter->AllocateTensors();
  inputTensor = tflInterpreter->input(0);
  outputTensor = tflInterpreter->output(0);

  // Initialize the camera
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = FRAMESIZE_QVGA;
  config.jpeg_quality = 12;
  config.fb_count = 1;

  // Check if the camera module initializes correctly
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x", err);
    return;
  }
}

void loop() {
  camera_fb_t *frameBuffer = esp_camera_fb_get();
  if (!frameBuffer) {
    Serial.println("Camera capture failed");
    return;
  }

  // Image processing and inference code here
  // This depends greatly on your model input requirements

  // Run the model on the input
  TfLiteStatus invokeStatus = tflInterpreter->Invoke();
  if (invokeStatus != kTfLiteOk) {
    Serial.println("Model invocation failed");
    esp_camera_fb_return(frameBuffer);
    return;
  }

  // Process the output from the model to detect human presence
  float* scores = outputTensor->data.f;
  int length = outputTensor->dims->data[outputTensor->dims->size - 1];
  for (int i = 0; i < length; i++) {
    if (scores[i] > 0.5) {  // Arbitrary threshold, adjust based on your model and needs
      digitalWrite(BUZZER_PIN, HIGH);
      delay(200);
      digitalWrite(BUZZER_PIN, LOW);
      break;
    }
  }

  esp_camera_fb_return(frameBuffer);
  delay(5000); // Delay between captures, adjust as needed
}
