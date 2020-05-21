
import time

for i in range(100):
   #print (f'process:       ', end='')
   print (f'\rprocess: {i}', end='')
   time.sleep(0.2)

