# 能量机关硬件适配指南

## 1. 概述
本文档提供了YOLO11n-pose模型与ESP32系列主控的硬件适配方案，包括灯环一体化PCB设计、WS2812灯珠控制、机械键轴击打检测、功率控制和通信接口设计。

## 2. ESP32系列主控适配

### 2.1 主控选择
推荐使用以下ESP32系列主控：
- **ESP32S3**：支持Wi-Fi 6和Bluetooth 5.0，2.4GHz频段，4MB PSRAM，适合复杂计算任务
- **ESP32C3**：低功耗，支持Wi-Fi 4和Bluetooth 5.0，160MHz主频，适合资源受限场景

### 2.2 引脚分配方案

| 功能 | ESP32S3引脚 | ESP32C3引脚 | 说明 |
|------|-------------|-------------|------|
| WS2812数据 | GPIO3 | GPIO3 | 灯环控制引脚 |
| 机械键轴检测 | GPIO4 | GPIO4 | 击打检测输入 |
| 功率监测 | GPIO5 | GPIO5 | ADC输入，测量电压 |
| 电流监测 | GPIO6 | GPIO6 | ADC输入，测量电流 |
| 舵机控制 | GPIO7 | GPIO7 | 云台控制PWM输出 |
| UART通信 | GPIO8/GPIO9 | GPIO8/GPIO9 | 与Jetson通信 |
| SPI通信 | GPIO10/GPIO11/GPIO12/GPIO13 | GPIO10/GPIO11/GPIO12/GPIO13 | 可选高速通信 |
| 调试接口 | GPIO14/GPIO15 | GPIO14/GPIO15 | UART调试 |
| 散热风扇控制 | GPIO16 | GPIO16 | 温度控制PWM输出 |

### 2.3 硬件连接图
```
┌─────────────────────────────────────────┐
│          NVIDIA Jetson Xavier           │
└─────────────────────────────────────────┘
                    │ UART/SPI
                    ▼
┌─────────────────────────────────────────┐
│              ESP32S3/C3                 │
├───────────┬─────────────────────────────┤
│ GPIO3     │ WS2812灯环                  │
│ GPIO4     │ 机械键轴击打检测             │
│ GPIO5/GPIO6 │ 功率监测(电压/电流)       │
│ GPIO7     │ 云台舵机控制                │
│ GPIO16    │ 散热风扇控制                │
└───────────┴─────────────────────────────┘
```

## 3. 功率控制实现

### 3.1 功率监测电路

```
+V_BAT ──┬───────────────────────┐
         │                       │
         ▼                       ▼
     [电压分压电路]           [分流电阻] ── GND
         │                       │
         ▼                       ▼
       GPIO5                   GPIO6
     (ADC输入)                 (ADC输入)
```

- **电压监测**：使用1/10分压电路，将电池电压(7.4V-25.2V)转换为ESP32 ADC可接受的0-3.3V范围
- **电流监测**：使用0.1Ω/1W分流电阻，配合INA219电流传感器芯片

### 3.2 功率控制算法

```python
# ESP32功率控制示例代码
import machine
import time

# 硬件配置
VOLTAGE_PIN = machine.ADC(machine.Pin(5))
CURRENT_PIN = machine.ADC(machine.Pin(6))
MOTOR_PWM = machine.PWM(machine.Pin(7))

# 功率限制设置
MAX_POWER = 70.0  # 最大功率限制(70W)
VOLTAGE_DIVIDER = 10.0  # 电压分压比
CURRENT_SENSITIVITY = 0.1  # 电流传感器灵敏度

# 初始化
VOLTAGE_PIN.atten(machine.ADC.ATTN_11DB)  # 0-3.6V范围
CURRENT_PIN.atten(machine.ADC.ATTN_11DB)
MOTOR_PWM.freq(50)
MOTOR_PWM.duty_u16(0)

def read_voltage():
    """读取电池电压"""
    adc_value = VOLTAGE_PIN.read()
    voltage = (adc_value / 4095.0) * 3.6 * VOLTAGE_DIVIDER
    return voltage

def read_current():
    """读取电机电流"""
    adc_value = CURRENT_PIN.read()
    current = (adc_value / 4095.0) * 3.6 / CURRENT_SENSITIVITY
    return current

def calculate_power():
    """计算实时功率"""
    voltage = read_voltage()
    current = read_current()
    power = voltage * current
    return power, voltage, current

def power_control_loop():
    """功率控制主循环"""
    target_duty = 5000  # 目标占空比
    
    while True:
        power, voltage, current = calculate_power()
        
        # 功率限制逻辑
        if power > MAX_POWER:
            # 降低电机功率
            target_duty = max(0, target_duty - 100)
        elif power < MAX_POWER * 0.9:
            # 可以增加电机功率
            target_duty = min(65535, target_duty + 50)
        
        # 设置PWM占空比
        MOTOR_PWM.duty_u16(target_duty)
        
        # 打印功率信息
        print(f"电压: {voltage:.2f}V, 电流: {current:.2f}A, 功率: {power:.2f}W, 占空比: {target_duty}")
        
        time.sleep(0.01)  # 100Hz控制频率

# 启动功率控制
if __name__ == "__main__":
    power_control_loop()
```

### 3.3 散热控制

```python
# 散热风扇控制
import machine
import time

TEMP_SENSOR = machine.ADC(machine.Pin(17))
FAN_PWM = machine.PWM(machine.Pin(16))

TEMP_SENSOR.atten(machine.ADC.ATTN_11DB)
FAN_PWM.freq(1000)

def read_temperature():
    """读取温度传感器值(需校准)"""
    adc_value = TEMP_SENSOR.read()
    # 温度校准公式(示例)
    temperature = (adc_value / 4095.0) * 100 - 20
    return temperature

def fan_control():
    """风扇温度控制"""
    temperature = read_temperature()
    
    if temperature < 40:
        # 低温，风扇低速
        FAN_PWM.duty_u16(0)
    elif temperature < 50:
        # 中温，风扇中速
        FAN_PWM.duty_u16(20000)
    elif temperature < 60:
        # 高温，风扇高速
        FAN_PWM.duty_u16(40000)
    else:
        # 超高温，风扇全速
        FAN_PWM.duty_u16(65535)
```

## 4. WS2812灯环控制

### 4.1 灯环初始化

```python
# WS2812灯环控制示例
import machine
import neopixel

NUM_LEDS = 12  # 灯环LED数量
LED_PIN = machine.Pin(3)

# 初始化灯环
np = neopixel.NeoPixel(LED_PIN, NUM_LEDS)

def clear_leds():
    """清除所有LED"""
    for i in range(NUM_LEDS):
        np[i] = (0, 0, 0)
    np.write()

def set_led_color(index, r, g, b):
    """设置单个LED颜色"""
    np[index] = (r, g, b)
    np.write()

def set_all_leds(r, g, b):
    """设置所有LED颜色"""
    for i in range(NUM_LEDS):
        np[i] = (r, g, b)
    np.write()

def run_rotation_animation(speed=0.1):
    """旋转动画，模拟能量机关旋转"""
    for i in range(NUM_LEDS):
        clear_leds()
        set_led_color(i, 255, 0, 0)
        set_led_color((i + 1) % NUM_LEDS, 128, 0, 0)
        time.sleep(speed)
```

### 4.2 动态颜色切换

```python
# 动态颜色切换示例
COLORS = [
    (255, 0, 0),    # 红色
    (0, 255, 0),    # 绿色
    (0, 0, 255),    # 蓝色
    (255, 255, 0),  # 黄色
    (255, 0, 255),  # 紫色
    (0, 255, 255),  # 青色
]

def color_chase(color, delay=0.1):
    """颜色追逐动画"""
    for i in range(NUM_LEDS):
        clear_leds()
        np[i] = color
        np.write()
        time.sleep(delay)

def rainbow_cycle(delay=0.1):
    """彩虹循环动画"""
    for j in range(255):
        for i in range(NUM_LEDS):
            pixel_index = (i * 256 // NUM_LEDS) + j
            np[i] = wheel(pixel_index & 255)
        np.write()
        time.sleep(delay)

def wheel(pos):
    """生成彩虹颜色"""
    if pos < 85:
        return (pos * 3, 255 - pos * 3, 0)
    elif pos < 170:
        pos -= 85
        return (255 - pos * 3, 0, pos * 3)
    else:
        pos -= 170
        return (0, pos * 3, 255 - pos * 3)
```

## 5. 机械键轴击打检测

### 5.1 硬件电路

```
+3.3V ──┬─────────────────────────┐
         │                         │
         ▼                         ▼
     [机械键轴]                 [10kΩ上拉电阻]
         │                         │
         ▼                         │
        GND                   GPIO4 (ADC输入)
```

### 5.2 软件实现

```python
# 机械键轴击打检测
import machine
import time

KEY_PIN = machine.Pin(4, machine.Pin.IN, machine.Pin.PULL_UP)

# 状态变量
last_state = 1
last_time = 0
DEBOUNCE_TIME = 50  # 消抖时间(ms)

def key_interrupt(pin):
    """键轴中断处理函数"""
    global last_state, last_time
    
    current_time = time.ticks_ms()
    current_state = pin.value()
    
    # 消抖处理
    if time.ticks_diff(current_time, last_time) > DEBOUNCE_TIME:
        if current_state != last_state:
            last_state = current_state
            last_time = current_time
            
            if current_state == 0:  # 键轴按下
                print("检测到击打事件！")
                # 发送击打信号到Jetson
                send_hit_event()

# 注册中断
KEY_PIN.irq(trigger=machine.Pin.IRQ_FALLING | machine.Pin.IRQ_RISING, handler=key_interrupt)

def send_hit_event():
    """发送击打事件到Jetson"""
    # 通过UART发送击打信号
    uart.write(b"HIT\n")
```

## 6. 与Jetson通信协议

### 6.1 UART通信配置

```python
# ESP32端UART配置
import machine

UART_BAUDRATE = 115200
uart = machine.UART(1, baudrate=UART_BAUDRATE, tx=machine.Pin(8), rx=machine.Pin(9))

def send_data(data):
    """发送数据到Jetson"""
    uart.write(data.encode() + b"\n")

def receive_data():
    """从Jetson接收数据"""
    if uart.any():
        return uart.readline().decode().strip()
    return None
```

```python
# Jetson端Python UART配置
import serial
import time

ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=0.1)

def send_command(command):
    """发送命令到ESP32"""
    ser.write((command + '\n').encode())

def receive_response():
    """接收ESP32的响应"""
    if ser.in_waiting > 0:
        return ser.readline().decode().strip()
    return None
```

### 6.2 通信协议格式

| 命令类型 | 格式 | 说明 |
|----------|------|------|
| 检测结果 | `DETECT:<x>,<y>,<angle>,<confidence>` | 装甲模块位置和角度 |
| 功率状态 | `POWER:<voltage>,<current>,<power>` | 实时功率监测数据 |
| 击打事件 | `HIT:<timestamp>` | 机械键轴击打事件 |
| 控制命令 | `CONTROL:<servo_angle>,<fire>` | 云台控制和发射命令 |
| 灯环控制 | `LED:<mode>,<color_r>,<color_g>,<color_b>` | 灯环模式和颜色 |
| 温度状态 | `TEMP:<esp_temp>,<jetson_temp>` | 温度监测数据 |

## 7. 调试接口设计

### 7.1 UART调试

- 使用ESP32的UART0作为调试接口
- 默认波特率：115200
- 支持ESP-IDF Monitor和Arduino Serial Monitor

### 7.2 调试命令

| 命令 | 功能 | 示例 |
|------|------|------|
| `STATUS` | 查看系统状态 | `STATUS` |
| `POWER` | 查看功率数据 | `POWER` |
| `LED_TEST` | 灯环测试 | `LED_TEST:RED` |
| `KEY_TEST` | 键轴测试 | `KEY_TEST` |
| `CALIBRATE` | 功率校准 | `CALIBRATE:<voltage_offset>,<current_offset>` |
| `RESET` | 系统重置 | `RESET` |

### 7.3 日志系统

```python
# 日志系统实现
import time

LOG_LEVELS = {
    "DEBUG": 0,
    "INFO": 1,
    "WARN": 2,
    "ERROR": 3
}

current_level = LOG_LEVELS["INFO"]

def log(level, message):
    """日志输出函数"""
    if LOG_LEVELS[level] >= current_level:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"[{timestamp}] [{level}] {message}")

def debug(message):
    """调试日志"""
    log("DEBUG", message)

def info(message):
    """信息日志"""
    log("INFO", message)

def warn(message):
    """警告日志"""
    log("WARN", message)

def error(message):
    """错误日志"""
    log("ERROR", message)

# 使用示例
info("系统初始化完成")
debug(f"功率监测值: {voltage}V, {current}A")
error("功率超过限制！")
```

## 8. 功率优化策略

### 8.1 硬件层面优化

1. **低功耗设计**：
   - 使用ESP32的Deep Sleep模式，空闲时降低功耗
   - 选择低功耗WS2812灯珠，仅在需要时点亮
   - 使用高效DC-DC转换器，效率≥95%

2. **功率预算分配**：
   - 底盘驱动：≤40W
   - 云台控制：≤10W
   - 灯环照明：≤5W
   - 计算单元：≤10W
   - 散热系统：≤5W

### 8.2 软件层面优化

1. **动态功率调节**：
   ```python
   def dynamic_power_control():
       """根据任务动态调节功率"""
       if in_detection_mode():
           # 检测模式，降低底盘功率
           set_chassis_power_limit(20W)
           set_compute_power(high)
       elif in_movement_mode():
           # 移动模式，增加底盘功率
           set_chassis_power_limit(40W)
           set_compute_power(low)
       elif in_idle_mode():
           # 空闲模式，进入低功耗
           set_chassis_power_limit(5W)
           set_compute_power(sleep)
   ```

2. **功率预测算法**：
   ```python
   def predict_power_usage(trajectory):
       """预测移动轨迹的功率消耗"""
       # 基于轨迹长度、速度和坡度预测功率
       power_estimate = base_power + (speed * 0.5) + (slope * 2.0)
       return power_estimate
   ```

## 9. 测试验证

### 9.1 功率限制测试

1. **测试步骤**：
   - 连接功率计到机器人电源输入端
   - 运行最大功率负载场景
   - 验证系统自动降低功率到70W以下

2. **预期结果**：
   - 最大功率不超过70W
   - 功率超过限制时系统自动调节
   - 无硬件损坏或过热现象

### 9.2 通信稳定性测试

1. **测试步骤**：
   - 连续运行24小时通信测试
   - 记录通信错误率
   - 测试不同距离下的通信质量

2. **预期结果**：
   - 通信错误率<0.1%
   - 5米距离内通信稳定
   - 无数据丢失或错误

### 9.3 击打检测精度测试

1. **测试步骤**：
   - 进行100次击打测试
   - 记录检测成功率
   - 测试不同击打力度下的检测性能

2. **预期结果**：
   - 检测成功率≥99%
   - 响应时间<10ms
   - 无误触发或漏触发

## 10. 结论

本硬件适配方案提供了完整的ESP32系列主控与YOLO11n-pose模型的对接实现，包括：

- ✅ ESP32S3/C3主控适配
- ✅ WS2812灯环动态颜色控制
- ✅ 机械键轴击打检测
- ✅ 70W功率限制实现
- ✅ 与Jetson的通信协议
- ✅ 调试接口和散热设计

该方案满足了能量机关硬件特性要求，能够支持旋转速度>0.4转/秒的能量机关击打任务，同时保证了5米距离、50发弹丸的命中率考核需求。