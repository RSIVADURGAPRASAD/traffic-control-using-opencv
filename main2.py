import cv2 as cv
import time
import RPi.GPIO as GPIO

# ---------------- LCD SETUP ----------------
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

# LCD Pins
LCD_RS = 26
LCD_E  = 19
LCD_D4 = 13
LCD_D5 = 6
LCD_D6 = 5
LCD_D7 = 11

# LED Pins
RED = 17
YELLOW = 27
GREEN = 22

GPIO.setup(LCD_E, GPIO.OUT)
GPIO.setup(LCD_RS, GPIO.OUT)
GPIO.setup(LCD_D4, GPIO.OUT)
GPIO.setup(LCD_D5, GPIO.OUT)
GPIO.setup(LCD_D6, GPIO.OUT)
GPIO.setup(LCD_D7, GPIO.OUT)

GPIO.setup(RED, GPIO.OUT)
GPIO.setup(YELLOW, GPIO.OUT)
GPIO.setup(GREEN, GPIO.OUT)

# LCD constants
LCD_WIDTH = 16
LCD_CHR = True
LCD_CMD = False
LCD_LINE_1 = 0x80
LCD_LINE_2 = 0xC0
E_PULSE = 0.0005
E_DELAY = 0.0005

def lcd_init():
    lcd_byte(0x33, LCD_CMD)
    lcd_byte(0x32, LCD_CMD)
    lcd_byte(0x06, LCD_CMD)
    lcd_byte(0x0C, LCD_CMD)
    lcd_byte(0x28, LCD_CMD)
    lcd_byte(0x01, LCD_CMD)
    time.sleep(E_DELAY)

def lcd_byte(bits, mode):
    GPIO.output(LCD_RS, mode)
    GPIO.output(LCD_D4, bits & 0x10 == 0x10)
    GPIO.output(LCD_D5, bits & 0x20 == 0x20)
    GPIO.output(LCD_D6, bits & 0x40 == 0x40)
    GPIO.output(LCD_D7, bits & 0x80 == 0x80)
    lcd_toggle_enable()

    GPIO.output(LCD_D4, bits & 0x01 == 0x01)
    GPIO.output(LCD_D5, bits & 0x02 == 0x02)
    GPIO.output(LCD_D6, bits & 0x04 == 0x04)
    GPIO.output(LCD_D7, bits & 0x08 == 0x08)
    lcd_toggle_enable()

def lcd_toggle_enable():
    time.sleep(E_DELAY)
    GPIO.output(LCD_E, True)
    time.sleep(E_PULSE)
    GPIO.output(LCD_E, False)
    time.sleep(E_DELAY)

def lcd_string(message, line):
    message = message.ljust(LCD_WIDTH, " ")
    lcd_byte(line, LCD_CMD)
    for i in range(LCD_WIDTH):
        lcd_byte(ord(message[i]), LCD_CHR)

# ---------------- YOLO SETUP ----------------
Conf_threshold = 0.4
NMS_threshold = 0.4
vehicle_classes = ["car", "motorbike", "bus", "truck", "bicycle"]

class_name = []
with open('classes.txt', 'r') as f:
    class_name = [cname.strip() for cname in f.readlines()]

net = cv.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')
net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

model = cv.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

#cap = cv.VideoCapture('output.avi')
cap = cv.VideoCapture(0)

# ---------------- TRAFFIC LIGHT CYCLE ----------------
def run_phase(duration, red, yellow, green, phase_name):
    """Run a traffic phase while keeping detection active."""
    start_time = time.time()
    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            break

        # Detection
        classes, scores, boxes = model.detect(frame, Conf_threshold, NMS_threshold)
        vehicle_count = 0
        for (classid, score, box) in zip(classes, scores, boxes):
            cname = class_name[classid]
            if cname in vehicle_classes:
                vehicle_count += 1
                cv.rectangle(frame, box, (0, 255, 0), 2)
                cv.putText(frame, cname, (box[0], box[1]-10),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Update LEDs
        GPIO.output(RED, red)
        GPIO.output(YELLOW, yellow)
        GPIO.output(GREEN, green)

        # LCD update
        lcd_string(f"{phase_name} LIGHT", LCD_LINE_1)
        lcd_string(f"Vehicles:{vehicle_count}", LCD_LINE_2)

        # Terminal log
        print(f"[{phase_name}] Vehicle Count: {vehicle_count}")

        # Show camera
        cv.imshow("Traffic Camera", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            return vehicle_count, False

    return vehicle_count, True

# ---------------- MAIN LOOP ----------------
lcd_init()
lcd_string("TRAFFIC SYSTEM", LCD_LINE_1)
lcd_string("STARTING...", LCD_LINE_2)
time.sleep(2)
vehicle_count=0
try:
    while True:
        # RED phase (5s, parallel detection)
        vehicle_count, ok = run_phase(5, 1, 0, 0, "RED")
        if not ok: break

        # YELLOW phase (2s, parallel detection)
        vehicle_count, ok = run_phase(2, 0, 1, 0, "YELLOW")
        if not ok: break

        # GREEN phase (5 + latest vehicle count)
        green_time = 5 + vehicle_count
        start_time = time.time()
        while time.time() - start_time < green_time:
            ret, frame = cap.read()
            if not ret:
                break

            # No need to update vehicle count during green → keep previous
            cv.putText(frame, "GREEN LIGHT", (30, 30),
                       cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            GPIO.output(RED, 0)
            GPIO.output(YELLOW, 0)
            GPIO.output(GREEN, 1)

            remaining = green_time - int(time.time() - start_time)
            lcd_string("GREEN LIGHT", LCD_LINE_1)
            lcd_string(f"Go:{remaining}s", LCD_LINE_2)
            print(f"[GREEN] {remaining}s left (vehicles={vehicle_count})")

            cv.imshow("Traffic Camera", frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                ok = False
                break

        if not ok:
            break

except KeyboardInterrupt:
    pass
finally:
    cap.release()
    cv.destroyAllWindows()
    GPIO.cleanup()
