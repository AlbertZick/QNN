# from network import Updater, HiddenLayer, ActiveFunct
import network as nw
from DataLoader import viewImage, DataLoader, Printer, DataConverter
from math import floor
import datetime
import numpy as np

today = datetime.datetime.today
now  = datetime.datetime.now

def getTime():
   date_time = str(today()) + '_' + str(now().time()).replace(':', '.')
   return date_time

def train(save=True):
   window_h = 3
   window_w = 3
   eval_delay = 5

   eval_err  = np.zeros((1,3,1), dtype=np.float64)
   train_err = np.zeros((1,3,1), dtype=np.float64)

   def _appendArray(arr, val, debug=False):
      if debug:
         print (val)
      if len(arr) == 1:
         arr[0] = val.reshape(3,1)
      else:
         # arr.append(val.reshape(1,3,1))
         arr = np.concatenate(val.reshape(1,3,1))

   up = nw.Updater(nw.Updater_enum.SGD, r=0.001)
   H1 = nw.HiddenLayer(9, 5, useBias=True)
   H2 = nw.HiddenLayer(5, 1, useBias=True)

   H1.compile(Update_c=up, ActFunc=nw.ActiveFunct_enum.sigm)
   H2.compile(Update_c=up, ActFunc=nw.ActiveFunct_enum.sigm)

   Loss = nw.LossFunc()

   trainSet = DataLoader("C:\\MyFolder\\MyData\\QNN\\QNN\\Sony\\Sony_train_list.txt", criteria='F10')
   evalSet  = DataLoader("C:\\MyFolder\\MyData\\QNN\\QNN\\Sony\\Sony_val_list.txt", criteria='F10')

   while (eval_delay > 0):
      # training
      # trainSet.restart(shuffle=True)
      trainSet.restart(shuffle=False)
      trainSet_d = trainSet.getNextImgData()
      img_err   = np.zeros((1,3,1), dtype=np.float64)
      while np.all(trainSet_d[0] != None):

         Printer.show(trainSet_d[-2], new=True)

         trainData  = DataConverter(X_data=trainSet_d[0], Y_data=trainSet_d[1], w=window_w, h=window_h)
         x_data, y_data = trainData.getNextMatrix()
         window_err  = np.zeros((1,3,1), dtype=np.float64)
         while np.all(x_data != None):

            Printer.show(f'Image ith: {trainSet.idx}/{trainSet.TotalImgs}, Process: {trainData.MatIdx}/{trainData.totalMats}', new=False)

            # forward
            H1_y = H1.forward(x_data)
            H2_y = H2.forward(H1_y)

            err  = Loss.calc(H2_y, y_data[floor(window_h/2), floor(window_w/2)])
            _appendArray(window_err, err)

            # back propagation
            loss = Loss.diff(H2_y, y_data[floor(window_h/2), floor(window_w/2)])
            DH2_x = H2.backprop(loss, H1_y)
            DH1_x = H1.backprop(DH2_x, x_data)

            # update
            H1.update()
            H2.update()

            # prepare Data for the next iteration
            x_data, y_data = trainData.getNextMatrix()

         _appendArray(img_err, np.average(window_err, axis=0), debug=False)

         trainSet_d = trainSet.getNextImgData()

      _appendArray(train_err, np.average(img_err, axis=0))


      # eval
      evalSet.restart(shuffle=True)
      eval_d   = evalSet.getNextImgData()
      eval_img_err   = np.zeros((1,3,1), dtype=np.float64)
      while np.all(eval_d[0] != None):
         evalData  = DataConverter(X_data=eval_d[0], Y_data=eval_d[1], w=window_w, h=window_h)
         x_data, y_data = evalData.getNextMatrix()
         
         eval_window_err = np.zeros((1,3,1), dtype=np.float64)
         while np.all(x_data != None):
            # forward
            H1_y = H1.forward(x_data)
            H2_y = H2.forward(H1_y)

            err  = Loss.calc(H2_y, y_data[floor(window_h/2), floor(window_w/2)])

            _appendArray(eval_window_err, err)

            # prepare Data for the next iteration
            x_data, y_data = evalData.getNextMatrix()

         _appendArray(eval_img_err, np.average(eval_window_err, axis=0))
         eval_d = trainSet.getNextImgData()

      _appendArray(eval_err, np.average(eval_img_err, axis=0))

      if eval_err[-1] > eval_err[-2]:
         eval_delay -= 1


   print ("Finish training")
   if save:
      date_time = getTime()
      H1.WrtModel(f'H1_{date_time}')
      H2.WrtModel(f'H2_{date_time}')

def test():
   pass

def main():
   train()


if __name__ == '__main__':
   main()