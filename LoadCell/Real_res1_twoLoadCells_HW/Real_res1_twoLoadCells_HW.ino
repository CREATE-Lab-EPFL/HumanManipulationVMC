// ============================================================
// Arduino Code: 2x Load Cells (HX711)
// 输出格式: Force1(g),Force2(g)
// ============================================================

#include "HX711.h"

// --- HX711 引脚定义 ---
// Load Cell 1: 水平剪切力
const int LOADCELL_DOUT_PIN_1 = 3;
const int LOADCELL_SCK_PIN_1  = 2;

// Load Cell 2: 垂直法向力
const int LOADCELL_DOUT_PIN_2 = 5;
const int LOADCELL_SCK_PIN_2  = 4;

// --- HX711 对象 ---
HX711 scale1;
HX711 scale2;

// --- 校准参数（需要你自己标定！） ---
float calibration_factor_1 = 1861.078002;  // 根据你的实际标定更新
float calibration_factor_2 = 2030.073927;

// --- 偏移量（tare）---
long offset_1 = 0;
long offset_2 = 0;

void setup() {
  Serial.begin(115200);
  Serial.println("Initializing 2x Load Cells...");

  // --- 初始化 HX711 ---
  scale1.begin(LOADCELL_DOUT_PIN_1, LOADCELL_SCK_PIN_1);
  scale2.begin(LOADCELL_DOUT_PIN_2, LOADCELL_SCK_PIN_2);

  delay(10);

  if (scale1.is_ready()) Serial.println("Load Cell 1 ready.");
  else Serial.println("Load Cell 1 not found!");

  if (scale2.is_ready()) Serial.println("Load Cell 2 ready.");
  else Serial.println("Load Cell 2 not found!");

  // --- 设置校准系数 ---
  scale1.set_scale(calibration_factor_1);
  scale2.set_scale(calibration_factor_2);

  // --- 去皮 ---
  Serial.println("Taring Load Cells...");
  offset_1 = scale1.read_average(10);
  offset_2 = scale2.read_average(10);

  scale1.set_offset(offset_1);
  scale2.set_offset(offset_2);

  Serial.println("Offsets set.");

  // --- 打印 CSV 表头 ---
  Serial.println("Force1(g),Force2(g)");
}

void loop() {
  // --- 读取 HX711 ---
  float force1 = scale1.get_units(1);
  float force2 = scale2.get_units(1);

  // 保留 4 位小数
  force1 = round(force1 * 10000.0) / 10000.0;
  force2 = round(force2 * 10000.0) / 10000.0;

  // --- 串口输出 CSV 格式 ---
  Serial.print(force1, 4);
  Serial.print(",");
  Serial.println(force2, 4);

  delay(10); // 100 Hz 采样
}
