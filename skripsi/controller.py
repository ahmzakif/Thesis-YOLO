import time
import serial
import threading
from adafruit_servokit import ServoKit

class Controller:
    def __init__(self, 
                 servo_pin=4, 
                 min_degree=0, 
                 max_degree=180, 
                 initial_degree=90,
                 serial_port='/dev/ttyAMA0',
                 baud_rate=9600):
        
        self.SERVO_PIN = servo_pin
        self.MIN_DEGREE = min_degree
        self.MAX_DEGREE = max_degree
        self.INITIAL_DEGREE = initial_degree
        
        self.turning = False
        
        self.kit = ServoKit(channels=16)
        self.kit.servo[self.SERVO_PIN].angle = self.INITIAL_DEGREE
        
        self.ser = serial.Serial(serial_port, baud_rate, timeout=1)
        time.sleep(2)  

    def __del__(self):
        if hasattr(self, 'ser') and self.ser.is_open:
            self.ser.close()

    def send_command(self, angle, message):
        self.turning = True
        self.kit.servo[self.SERVO_PIN].angle = angle
        if self.ser.is_open:
            self.ser.write(message.encode())
        time.sleep(0.5)
        self.turning = False

    def send_message(self, message):
        if self.ser.is_open:
            self.ser.write(message.encode())

    def servo_control(self, object_type):
        if not self.turning:
            if object_type.lower() == 'metal':
                threading.Thread(
                    target=self.send_command,
                    args=(self.MIN_DEGREE, 'M'),
                    daemon=True
                ).start()
                
            elif object_type.lower() == 'plastic':
                threading.Thread(
                    target=self.send_command,
                    args=(self.MAX_DEGREE, 'P'),
                    daemon=True
                ).start()

    def servo_reset(self):
        if not self.turning:
            threading.Thread(
                target=self.send_command,
                args=(self.INITIAL_DEGREE, 'R'),
                daemon=True
            ).start()

    def close(self):
        if self.ser.is_open:
            self.ser.close()