from network import Updater, HiddenLayer, ActiveFunct_enum, Updater_enum, LossFunc
from DataLoader import viewImage, DataLoader, Printer, DataConverter
from math import floor
import datetime
import numpy as np
import sys, os
import gc


today = datetime.datetime.today
now   = datetime.datetime.now


def getTime():
   # date_time = str(today()) + '_' + str(now().time()).replace(':', '.')
   date_time = str(today()).replace(':', '_').replace(' ', '_').replace('.', '_')
   return date_time

def train(trainFile, evalFile, testFile, initFile, save=True):
   Prt = Printer(logFile=f'log_{getTime()}.log')

   window_h   = 3
   window_w   = 3
   eval_delay = 5

   eval_cntr = 1

   eval_err  = np.array([None])
   train_err = np.array([None])

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
   up = Updater(Updater_enum.SGD, r=0.001)
   H1 = HiddenLayer(9, 5, useBias=True)
   H2 = HiddenLayer(5, 1, useBias=True)

   H1.compile(Update_c=up, ActFunc=ActiveFunct_enum.sigm)
   H2.compile(Update_c=up, ActFunc=ActiveFunct_enum.sigm)

   Loss = LossFunc()


   # Get data
   trainSet = DataLoader(trainFile, criteria='F10')
   evalSet  = DataLoader(evalFile, criteria='F10')

   while (eval_delay > 0):
      # training
      Prt.show (f'Start training model , iteration={eval_cntr}', new=True)
      # trainSet.restart(shuffle=True)
      trainSet.restart(shuffle=False)
      trainSet_d = trainSet.getNextImgData()
      img_err   = np.array([None])
      while np.all(trainSet_d[0] != None):
         Prt.show(trainSet_d[-2] + str(f'--*--ImageSize= {len(trainSet_d[0])} x {len(trainSet_d[0][0])}'), new=True)

         trainData  = DataConverter(X_data=trainSet_d[0], Y_data=trainSet_d[1], w=window_w, h=window_h)
         del trainSet_d

         ## Test code ################################################################################################################
         trainData.max_r_ptr = 4
         trainData.max_c_ptr = 4
         ##############################################################################################################################
         
         x_data, y_data = trainData.getNextMatrix()
         window_err  = np.array([None])
         while np.all(x_data != None):

            Prt.show(f'--> Image ith: {trainSet.idx}/{trainSet.TotalImgs}, '+\
                         f'Process: {trainData.MatIdx}/{trainData.totalMats}'+\
                                    f'~~({trainData.MatIdx/trainData.totalMats*100:4.0f}%)', new=False)

            # forward
            H1_y = H1.forward(x_data)
            H2_y = H2.forward(H1_y)


            y_data_sl = np.array(y_data[floor(window_h*window_w/2)], dtype=np.float64).reshape(1,3,1)
            err  = Loss.calc(H2_y, y_data_sl)
            window_err = _appendArray(window_err, err, debug=False)

            # back propagation
            loss = Loss.diff(H2_y, y_data_sl)
            DH2_x = H2.backprop(loss, H1_y)
            DH1_x = H1.backprop(DH2_x, x_data)

            # update
            H1.update()
            H2.update()

            ## Test code ################################################################################################################
            # Prt.show(f'H2_y={H2_y}, H1_y={H1_y}, x_data={x_data[0]}', new=True)
            # Prt.show(f'loss={loss}, y_data={y_data_sl}, H2_y={H2_y}')
            # Prt.show(f'DH2_x={DH2_x}, DH1_x={DH2_x}')
            # os._exit(1)
            # os.system("pause")
            ##############################################################################################################################

            # prepare Data for the next iteration
            x_data, y_data = trainData.getNextMatrix()


            ##############################################################################################################################
            ## Test code ################################################################################################################
            gc.collect()
            ##############################################################################################################################

         img_err = _appendArray(img_err, np.average(window_err, axis=0), debug=False)

         trainSet_d = trainSet.getNextImgData()

      train_err = _appendArray(train_err, np.average(img_err, axis=0), debug=False)
      Prt.show (f'img_err={img_err}', new=True)


      ########## EVALUATING ############################
      Prt.show (f'Start evaluating model , iteration={eval_cntr}', new=True)
      eval_cntr += 1
      # eval
      evalSet.restart(shuffle=True)
      eval_d   = evalSet.getNextImgData()
      eval_img_err   = np.array([None])
      while np.all(eval_d[0] != None):
         Prt.show(eval_d[-2] + str(f'Size: {len(eval_d[0])}, {len(eval_d[0][0])}'), new=True)

         evalData  = DataConverter(X_data=eval_d[0], Y_data=eval_d[1], w=window_w, h=window_h)
         
         ## Test code ################################################################################################################
         evalData.max_r_ptr = 4
         evalData.max_c_ptr = 4
         ##############################################################################################################################

         x_data, y_data = evalData.getNextMatrix()
         
         eval_window_err = np.array([None])
         while np.all(x_data != None):
            Prt.show(f'Image ith: {evalSet.idx}/{evalSet.TotalImgs}, '+\
                         f'Process: {evalData.MatIdx}/{evalData.totalMats}'+\
                                    f'~~({evalData.MatIdx/evalData.totalMats*100:4.0f}%)', new=False)

            # forward
            H1_y = H1.forward(x_data)
            H2_y = H2.forward(H1_y)

            err  = Loss.calc(H2_y, y_data[floor(window_h/2), floor(window_w/2)])

            eval_window_err = _appendArray(eval_window_err, err)

            # prepare Data for the next iteration
            x_data, y_data = evalData.getNextMatrix()

         eval_img_err = _appendArray(eval_img_err, np.average(eval_window_err, axis=0))
         eval_d = trainSet.getNextImgData()

      # Prt.show (f'eval_img_err={eval_img_err}', new=True)
      eval_err = _appendArray(eval_err, np.average(eval_img_err, axis=0), debug=False)
      Prt.show (f'eval_err={eval_err}', new=True)

      if len(eval_err)>1 and np.all(eval_err[-1] > eval_err[-2]):
         eval_delay -= 1


   print ("Finish training!!")
   if save:
      date_time = getTime()
      H1.WrtModel(f'H1_{date_time}')
      H2.WrtModel(f'H2_{date_time}')

def test():
   pass

def main():
   try:
      args = sys.argv[1:]
   except:
      args = []

   # Parse input args
   if 'test' in args:
      test(save=True)
      return
   if 'rtrain' in args:
      rtrain(save=True)
      return
   if 'train' in args:
      if '-tr' in args:
         TrainDataFile = args[args.index('-tr') + 1]
      else:
         TrainDataFile = 'train_list.txt'
      if '-v' in args:
         EvalDataFile  = args[args.index('-v') + 1]
      else:
         EvalDataFile  = 'val_list.txt'
      if '-t' in args:
         TestDataFile  = args[args.index('-t') + 1]
      else:
         TestDataFile  = 'test_list.txt'
      if '-i' in args:
         InitFile      = args[args.index('-i') + 1]
      else:
         InitFile      = 'initParams.txt'

      train(TrainDataFile, EvalDataFile, TestDataFile, InitFile, save=True)


if __name__ == '__main__':
   main()