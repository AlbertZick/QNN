
import time

for i in range(100):
   print (f'process:       ', end='\r')
   print (f'process: {i}', end='\r')
   time.sleep(0.2)

