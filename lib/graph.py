import matplotlib.pyplot as plt
import matplotlib.animation as animation
import threading
import time
import rawpy
import os


class Printer:
   def __init__(self, logFile='log.log', max_len=120):
      self.max_len    = max_len
      self.logFile    = logFile
      self.crtlogFile = True
      self.nextLine   = True

   def show(self, data, new=True):
      # print (''.join([' ']*self.max_len), end='\r')
      String = str(data)
      if new:
         if self.nextLine:
            print('',end='\n')
            self.nextLine = False

         print(String, end='\n')
         if self.crtlogFile:
            file = open(self.logFile, 'w')
            file.write(String+'\n')
            file.close()
            self.crtlogFile = False
         else:
            file = open(self.logFile, 'a')
            file.write(String+'\n')
            file.close()

      else:
         print (String + ''.join([' ']*(self.max_len-len(String))), end='\r')
         self.nextLine = True


def viewImage(data, save=False):
   fig  = plt.figure()
   plot = fig.add_subplot(1, 2, 1)
   plot.imshow(data[0])
   plot.set_title('X image')

   plot = fig.add_subplot(1, 2, 2)
   plot.imshow(data[1])
   plot.set_title('Y image')
   if save:
      name, _ = os.path.splitext(data[2].split('/')[-1])
      name = name + '.png'
      fig.savefig(name)

   fig.show()
   plt.show()



def animate(i, fig, Subplot, Diction):
   Subplot.clear()

   listColor = ['red', 'green', 'blue', 'magenta']
   color_cntr = 0

   # if len(Diction.keys()) == 4:
   for key in Diction.keys():
      if key == 'Running':
         continue
      Subplot.plot(range(len( Diction[key] )), Diction[key], color=listColor[color_cntr], label=key)
      color_cntr += 1

   Subplot.legend()
   fig.canvas.draw()

def animateGraph(Diction):
   fig = plt.figure(figsize=(10, 4))
   Subplot = fig.add_subplot(1, 1, 1)
   
   fig.show()
   Subplot.clear()

   ani = animation.FuncAnimation(fig, func=animate, interval=1000, fargs=(fig, Subplot, Diction))
   plt.show()
   # plt.draw()
   # while Diction['Running']:
   #    print ('This thread')
   #    time.sleep(1)

class Graph:
   def __init__(self, DataDict):
      self.DataDict = DataDict

      self.crtGraph()

   def crtGraph(self):
      self.Thr = threading.Thread(target=animateGraph, args=(self.DataDict,))
      self.Thr.start()



def main():
   import matplotlib

   from random import random
   Diction = {"r_data":[], 'g_data':[], 'b_data':[], 'Overall': [], "Running":True}

   grap = Graph(Diction)

   for i in range(20):
      Diction['r_data'].extend([10*random(),])
      Diction['g_data'].extend([10*random(),])
      Diction['b_data'].extend([10*random(),])
      Diction['Overall'].extend([10*random(),])
      print ('Main thread')
      time.sleep(0.5)

   Diction["Running"] = False



if __name__ == '__main__':
   main()
