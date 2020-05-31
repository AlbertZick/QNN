from numpy import sqrt as Sqrt
import keyboard
import time
import threading

a,b,c,d,x,y,z=1,2,3,4,5,6,7



R = ((a**2+b**2-c**2-d**2)*x + 2*y*(b*c-d*a) + 2*z*(b*d+c*a))/Sqrt(a**2+b**2+c**2+d**2)
G = ((a**2+c**2-d**2-b**2)*y + 2*x*(b*c+d*a) + 2*z*(c*d-b*a))/Sqrt(a**2+b**2+c**2+d**2)
B = ((a**2+d**2-b**2-c**2)*z + 2*x*(b*d-c*a) + 2*y*(c*d+b*a))/Sqrt(a**2+b**2+c**2+d**2)



# =====>**Da
dR_da=((a**3*x+2*(b**2+c**2+d**2)*(-d*y+c*z)+a*(b**2*x+3*(c**2+d**2)*x-2*b*(c*y+d*z)))/(a**2+b**2+c**2+d**2)**(3/2))
dG_da=((a**3*y-2*(b**2+c**2+d**2)*(-d*x+b*z)+a*(-2*b*c*x+3*b**2*y+c**2*y+3*d**2*y-2*c*d*z))/(a**2+b**2+c**2+d**2)**(3/2))
dB_da=((1/((a**2+b**2+c**2+d**2)**(3/2)))*(-2*c**3*x-2*a*b*d*x+2*b**3*y+2*b*(c**2+d**2)*y-2*c*d*(d*x+a*y)+3*a*c**2*z+a*(a**2+d**2)*z+b**2*(-2*c*x+3*a*z)))

# =====>**Db
dR_db=((b**3*x+3*b*(c**2+d**2)*x+2*a*b*(d*y-c*z)+2*(c**2+d**2)*(c*y+d*z)+a**2*(b*x+2*c*y+2*d*z))/(a**2+b**2+c**2+d**2)**(3/2))
dG_db=((1/((a**2+b**2+c**2+d**2)**(3/2)))*(2*c**3*x-3*b*c**2*y-b*(b**2+d**2)*y+a**2*(2*c*x-3*b*y)-2*a**3*z+2*c*d*(d*x-b*z)-2*a*(b*d*x+(c**2+d**2)*z)))
dB_db=((1/((a**2+b**2+c**2+d**2)**(3/2)))*(2*d**3*x+2*a**3*y-2*b*c*d*y+2*a*(b*c*x+(c**2+d**2)*y)-b**3*z-3*b*d**2*z+a**2*(2*d*x-3*b*z)+c**2*(2*d*x-b*z)))

# =====>**Dc
dR_dc=((1/((a**2+b**2+c**2+d**2)**(3/2)))*(-3*b**2*c*x-c*(c**2+d**2)*x+2*b**3*y+a**2*(-3*c*x+2*b*y)+2*a**3*z+2*b*d*(d*y-c*z)+2*a*(c*d*y+(b**2+d**2)*z)))
dG_dc=((1/((a**2+b**2+c**2+d**2)**(3/2)))*(2*b**3*x+2*b*d**2*x+c**3*y+3*c*d**2*y+2*d**3*z+a*c*(-2*d*x+2*b*z)+a**2*(2*b*x+c*y+2*d*z)+b**2*(3*c*y+2*d*z)))
dB_dc=((1/((a**2+b**2+c**2+d**2)**(3/2)))*(-2*(a**2+b**2+c**2+d**2)*(a*x-d*y+c*z)-c*(2*(-a*c+b*d)*x+2*(a*b+c*d)*y+(a**2-b**2-c**2+d**2)*z)))

# =====>**Dd
dR_dd=((1/((a**2+b**2+c**2+d**2)**(3/2)))*(-3*b**2*d*x-d*(c**2+d**2)*x-2*a**3*y+2*b**3*z+a**2*(-3*d*x+2*b*z)+2*b*c*(-d*y+c*z)-2*a*(b**2*y+c*(c*y+d*z))))
dG_dd=((1/((a**2+b**2+c**2+d**2)**(3/2)))*(2*(a**2+b**2+c**2+d**2)*(a*x-d*y+c*z)-d*(2*(b*c+a*d)*x+(a**2-b**2+c**2-d**2)*y+2*(-a*b+c*d)*z)))
dB_dd=((1/((a**2+b**2+c**2+d**2)**(3/2)))*(2*b**3*x+2*b*c**2*x+2*c**3*y+2*a*d*(c*x-b*y)+3*c**2*d*z+d**3*z+a**2*(2*b*x+2*c*y+d*z)+b**2*(2*c*y+3*d*z)))

# print ( 
#    f'{dR_da} '+\
#    f'{dG_da} '+\
#    f'{dB_da} '+\
#    f'{dR_db} '+\
#    f'{dG_db} '+\
#    f'{dB_db} '+\
#    f'{dR_dc} '+\
#    f'{dG_dc} '+\
#    f'{dB_dc} '+\
#    f'{dR_dd} '+\
#    f'{dG_dd} '+\
#    f'{dB_dd} ' )
#    
from lib.DataLoader import loadModel
from lib.network    import trigonometric_QCNN, Updater, Updater_enum, ActiveFunct_enum, LossFunc_enum, LossFunc
from lib.graph      import viewImage, Printer
from lib.DataLoader import DataLoader, DataConverter
import matplotlib.pyplot as plt
from   math import floor
import keyboard
import datetime
import numpy as np
import sys, os
import gc

def main_0():

   Prt = Printer(logFile=f'log_test_data.log')

   testFile = "C:\\MyFolder\\MyData\\QNN\\QNN\\test_list.txt"

   Dict = loadModel('C:\\MyFolder\\MyData\\QNN\\QNN\\model_2020-05-29_18_04_29_393142.txt')

   window_h   = 6
   window_w   = 6
   eval_delay = 5

   eval_cntr  = 0

   train_cntr = 0

   write_cntr = 0


   def _appendArray(arr, val, debug=False):
      if debug:
         print (val)
      if np.all(arr == None):
         arr = val.reshape(1,3,1)
      else:
         # arr.append(val.reshape(1,3,1))
         arr = np.concatenate((arr, val.reshape(1,3,1)), axis=0)

      return arr

   # create model
   up = Updater(Updater_enum.SGD, r=0.01)

   H1 = trigonometric_QCNN(window_h*window_w, (window_h*window_w)*12, 1, useBias=True)
   H2 = trigonometric_QCNN((window_h*window_w)*12, window_h*window_w, 1, useBias=True)

   H1.connect(H2)

   H1.compile(Update_c=up, ActFunc=ActiveFunct_enum.sigm)
   H2.compile(Update_c=up, ActFunc=ActiveFunct_enum.sigm)

   H1.loadModel(Dict[0])
   H2.loadModel(Dict[1])

   Loss = LossFunc()

   testSet = DataLoader(testFile, criteria='F10')
   testSet.restart(shuffle=True)
   testSet_d = testSet.getNextImgData(debug=True)

   img_err   = np.array([None])
   trainData  = DataConverter(X_data=testSet_d[0], Y_data=testSet_d[1], w=window_w, h=window_h)


   size_r, size_c = len(testSet_d[0]), len(testSet_d[0][0])

   r_low = floor(floor(size_r/3)/window_h)
   c_low = floor(floor(size_c/3)/window_w)

   trainData.setWindowRange(c_low=c_low*window_w, c_high=c_low*window_w + window_w*50, r_low=r_low*window_h, r_high=r_low*window_h + window_h*50, debug=True )

   x_data, y_data = trainData.getWindowMatDataInRange()

   ## Test code ################################################################################################################
   trainData.max_r_ptr = window_h*5
   trainData.max_c_ptr = window_w*5
   ##############################################################################################################################

   # predict_Image = np.zeros(trainData.Y_data.shape, dtype=np.float64)
   predict_Image = np.zeros(trainData.getPartImageShape(), dtype=np.float64)
   real_Image = np.zeros(trainData.getPartImageShape(), dtype=np.float64)

   window_err  = np.array([None])
   cntr = 0

   err = None
   while np.all(x_data != None):

      ##############################################################################################################################
      ## Test code ################################################################################################################
      gc.collect()
      cntr += 1
      Prt.show(f'Process: {cntr}/{trainData.total_range_mat}'+\
                  f' error={err}'
               )
      ##############################################################################################################################

      # forward
      H1_y = H1.forward(x_data)
      H2_y = H2.forward(H1_y)

      err  = Loss.calc(H2_y, y_data)
      trainData.convertPartImage(predict_Image, H2_y,debug=True)
      trainData.convertPartImage(real_Image, y_data)

      err  = np.average(err, axis=0)


      window_err = _appendArray(window_err, err, debug=False)

      # prepare Data for the next iteration
      x_data, y_data = trainData.getWindowMatDataInRange()


   img_err = _appendArray(img_err, np.average(window_err, axis=0), debug=False)

   predict_Image = predict_Image * 255
   predict_Image = predict_Image.astype(int)

   real_Image = real_Image * 255
   real_Image = real_Image.astype(int)

   print(real_Image)


   fig  = plt.figure()
   plot = fig.add_subplot(2, 2, 3)
   plot.imshow(predict_Image.tolist())
   plot.set_title('predicted image')

   plot = fig.add_subplot(2, 2, 1)
   plot.imshow(testSet_d[1])
   plot.set_title('Y image')

   plot = fig.add_subplot(2, 2, 4)
   plot.imshow(real_Image.tolist())
   plot.set_title('part of Y image')

   save = True
   if save:
      # name, _ = os.path.splitext(data[2].split('/')[-1])
      # name = name + '.png'
      fig.savefig('test.png')

   fig.show()
   plt.show()


def main_1():
   testSet = DataLoader(testFile, criteria='F10')
   testSet.restart(shuffle=True)
   testSet_d = testSet.getNextImgData(debug=True)

   img_err   = np.array([None])
   trainData  = DataConverter(X_data=testSet_d[0], Y_data=testSet_d[1], w=window_w, h=window_h)


   size_r, size_c = len(testSet_d[0]), len(testSet_d[0][0])
   r_low = floor(floor(size_r/3)/window_h)
   c_low = floor(floor(size_c/3)/window_w)
   trainData.setWindowRange(c_low=0, c_high=c_low*window_w, r_low=r_low*window_h, r_high=r_low*window_h, debug=True )
   x_data, y_data = trainData.getWindowMatDataInRange()



if __name__ == '__main__':
   main_0()

