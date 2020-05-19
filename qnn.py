from network import Updater, HiddenLayer, ActiveFunct
from DataLoader import viewImage, DataLoader


def main():

   eval_delay = 5
   eval_err  = []
   check_err = []


   up = Updater(Updater.SGD, r=0.001)
   H1 = HiddenLayer(9, 5, useBias=True)
   H2 = HiddenLayer(5, 1, useBias=True)

   H1.compile(Update_c=up, ActFunc=ActiveFunct.sigm)
   H2.compile(Update_c=up, ActFunc=ActiveFunct.sigm)


   while (eval_delay > 0):

      # training
      trainSet   = DataLoader("C:\\MyFolder\\MyData\\QNN\\QNN\\Sony\\Sony_train_list.txt")
      trainSet_d = trainSet.getNextImgData()
      while (bool(trainSet_d[0])):

         trainData  = DataConverter(X_data=, Y_data=, w=, h=)








      # eval


      evalSet    = DataLoader("C:\\MyFolder\\MyData\\QNN\\QNN\\Sony\\Sony_eval_list.txt")
      evalData   = None








if __name__ == '__main__':
   main()