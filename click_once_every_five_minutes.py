from pynput.mouse import Controller, Button
import time
import datetime

mouse = Controller()

while True:
    mouse.click(Button.left, 1)
    print('Clicked once at {}'.format(datetime.datetime.now()))

    time.sleep(300)