import datetime
import time

a = datetime.datetime.now()
time.sleep(3)
b = datetime.datetime.now()

c = b - a

print(c.microseconds)