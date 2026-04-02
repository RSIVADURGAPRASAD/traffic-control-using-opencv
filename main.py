import cv2 as cv
import time
import RPi.GPIO as GPIO  # If on Raspberry Pi (for LEDs)

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

LCD_RS = 26
LCD_E  = 19
LCD_D4 = 13
LCD_D5 = 6
LCD_D6 = 5
LCD_D7 = 11

GPIO.setup(LCD_E, GPIO.OUT)  # E
GPIO.setup(LCD_RS, GPIO.OUT) # RS
GPIO.setup(LCD_D4, GPIO.OUT) # DB4
GPIO.setup(LCD_D5, GPIO.OUT) # DB5
GPIO.setup(LCD_D6, GPIO.OUT) # DB6
GPIO.setup(LCD_D7, GPIO.OUT) # DB7

# Define some device constants
LCD_WIDTH = 16    # Maximum characters per line
LCD_CHR = True
LCD_CMD = False

LCD_LINE_1 = 0x80 # LCD RAM address for the 1st line
LCD_LINE_2 = 0xC0 # LCD RAM address for the 2nd line

# Timing constants
E_PULSE = 0.0005
E_DELAY = 0.0005


def lcd_init():
  # Initialise display
  lcd_byte(0x33,LCD_CMD) # 110011 Initialise
  lcd_byte(0x32,LCD_CMD) # 110010 Initialise
  lcd_byte(0x06,LCD_CMD) # 000110 Cursor move direction
  lcd_byte(0x0C,LCD_CMD) # 001100 Display On,Cursor Off, Blink Off
  lcd_byte(0x28,LCD_CMD) # 101000 Data length, number of lines, font size
  lcd_byte(0x01,LCD_CMD) # 000001 Clear display
  time.sleep(E_DELAY)

def lcd_byte(bits, mode):
  # Send byte to data pins
  # bits = data
  # mode = True  for character
  #        False for command

  GPIO.output(LCD_RS, mode) # RS

  # High bits
  GPIO.output(LCD_D4, False)
  GPIO.output(LCD_D5, False)
  GPIO.output(LCD_D6, False)
  GPIO.output(LCD_D7, False)
  if bits&0x10==0x10:
    GPIO.output(LCD_D4, True)
  if bits&0x20==0x20:
    GPIO.output(LCD_D5, True)
  if bits&0x40==0x40:
    GPIO.output(LCD_D6, True)
  if bits&0x80==0x80:
    GPIO.output(LCD_D7, True)

  # Toggle 'Enable' pin
  lcd_toggle_enable()

  # Low bits
  GPIO.output(LCD_D4, False)
  GPIO.output(LCD_D5, False)
  GPIO.output(LCD_D6, False)
  GPIO.output(LCD_D7, False)
  if bits&0x01==0x01:
    GPIO.output(LCD_D4, True)
  if bits&0x02==0x02:
    GPIO.output(LCD_D5, True)
  if bits&0x04==0x04:
    GPIO.output(LCD_D6, True)
  if bits&0x08==0x08:
    GPIO.output(LCD_D7, True)

  # Toggle 'Enable' pin
  lcd_toggle_enable()

def lcd_toggle_enable():
  # Toggle enable
  time.sleep(E_DELAY)
  GPIO.output(LCD_E, True)
  time.sleep(E_PULSE)
  GPIO.output(LCD_E, False)
  time.sleep(E_DELAY)

def lcd_string(message,line):
  # Send string to display




  message = message.ljust(LCD_WIDTH," ")

  lcd_byte(line, LCD_CMD)

  for i in range(LCD_WIDTH):
    lcd_byte(ord(message[i]),LCD_CHR)



lcd_init()
lcd_byte(0x01,LCD_CMD)
lcd_string("   WELCOME",LCD_LINE_1)


# -------------------- CONFIG --------------------
Conf_threshold = 0.4
NMS_threshold = 0.4
COLORS = [(0, 255, 0), (0, 0, 255), (255, 0, 0),
          (255, 255, 0), (255, 0, 255), (0, 255, 255)]

# Vehicle classes from COCO dataset
vehicle_classes = ["car", "motorbike", "bus", "truck", "bicycle"]

# GPIO setup
RED = 17
YELLOW = 27
GREEN = 22
GPIO.setmode(GPIO.BCM)
GPIO.setup(RED, GPIO.OUT)
GPIO.setup(YELLOW, GPIO.OUT)
GPIO.setup(GREEN, GPIO.OUT)

# -------------------- YOLO INIT --------------------
class_name = []
with open('classes.txt', 'r') as f:
    class_name = [cname.strip() for cname in f.readlines()]

net = cv.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')
net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

model = cv.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

cap = cv.VideoCapture('output.avi')

# -------------------- FUNCTIONS --------------------
def get_vehicle_count(frame):
    """Detect vehicles in frame and return count."""
    classes, scores, boxes = model.detect(frame, Conf_threshold, NMS_threshold)
    count = 0
    for (classid, score, box) in zip(classes, scores, boxes):
        cname = class_name[classid]
        if cname in vehicle_classes:  # Only vehicles
            count += 1
            color = COLORS[int(classid) % len(COLORS)]
            label = "%s : %.2f" % (cname, score)
            cv.rectangle(frame, box, color, 1)
            cv.putText(frame, label, (box[0], box[1]-10),
                       cv.FONT_HERSHEY_COMPLEX, 0.4, color, 1)
    return count, frame

def traffic_signal(vehicle_count):
    """Traffic light logic based on vehicle count."""

    # RED phase
    GPIO.output(RED, 1); GPIO.output(YELLOW, 0); GPIO.output(GREEN, 0)
    print("RED - Stop (5s)")
    time.sleep(5)

    # YELLOW phase
    GPIO.output(RED, 0); GPIO.output(YELLOW, 1); GPIO.output(GREEN, 0)
    print("YELLOW - Ready (2s)")
    time.sleep(2)

    # GREEN phase
    green_time = 5 + vehicle_count
    GPIO.output(RED, 0); GPIO.output(YELLOW, 0); GPIO.output(GREEN, 1)
    print(f"GREEN - Go ({green_time}s, vehicle count = {vehicle_count})")
    for i in range(green_time, 0, -1):
        print(f"Green light countdown: {i} sec", end="\r")
        time.sleep(1)

    GPIO.output(GREEN, 0)  # Turn off after cycle

# -------------------- MAIN LOOP --------------------
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        vehicle_count, frame = get_vehicle_count(frame)
        print(f"Detected Vehicles: {vehicle_count}")

        # Show camera window
        cv.imshow("Traffic Camera", frame)

        # Run one traffic signal cycle
        traffic_signal(vehicle_count)

        key = cv.waitKey(1)
        if key == ord('q'):
            break

except KeyboardInterrupt:
    pass

finally:
    cap.release()
    cv.destroyAllWindows()
    GPIO.cleanup()
