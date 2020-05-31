"""
Class DataLoader:

Return a specific matrix to train in each step

   ==== column ====>
 ||
 ||
row                         [ row_num, col_num ]
 ||
 ||
 \/

Initial:
   + get list of training images from a input file
   + get square window size
   + get slide step, default step equals to the window size
   

Method
   + get number of matrices. if the number of returned matrices is equal to the max, return None, otherwise, return a random matrix for each step
   + return a sequencial matrix for each step
   + Give a range and return matrix within that range, random or sequencial


"""









import numpy as np
import rawpy
import matplotlib.pyplot as plt
import os
import random
import json
from math import floor


def WrtModel(LstModel, fileName):
   WrtData = []
   for x in LstModel:
      WrtData.append(x.model2Dict())


   print(f'Writing model with {fileName}')

   file = open(fileName, 'w')
   file.write( json.dumps(WrtData))
   file.close()


def loadModel(fileName):
   Rst = []
   file = open (fileName, 'r')
   Rst = json.load(file)
   # print (Rst)
   return Rst



class DataLoader:
   def __init__(self, ListFile, **kwargs):
      self.img_idx = 0




      self.win_size = kwargs['win_size']
      self.slide    = kwargs['slide']
      
      self.imageSize = [-1, -1]

      self.r_lo = 0
      self.c_lo = 0

      self.r_hi = -1
      self.c_hi = -1


      self.max_mat  = -1
      self.mat_cntr = -1


   def initImageList(self, ListFile, criteria=None, shuffle=False):
      self._imageList = open(ListFile, 'r').readlines()
      self._imageList = [ line for line in self.Files if '#' not in line ]

      # only select some images has the same Focus
      if bool(criteria):
         self._criteria = criteria
         self._imageList = [ f for f in self.Files if self._criteria in f]

      if shuffle:
         self.shuffleImages()

   def getNextImage(self, debug=False):
      while self.img_idx < len(self._imageList):
         line = self._imageList[self.img_idx]
         self.img_idx += 1

         x_file, y_file, Sensity, F = [ i for i in line.split(' ') if i != '' ]

         if debug:
            print(f'x_file=[{x_file}]')
            print(f'y_file=[{y_file}]')

         x_data = rawpy.imread(x_file).postprocess(use_camera_wb=True)
         y_data = rawpy.imread(y_file).postprocess(use_camera_wb=True)
         if len(x_data[0][0]) != 3:
            print(f'Error: image {x_file} format is in RGBA, not in RGB')
            continue
         if len(y_data[0][0]) != 3:
            print(f'Error: image {y_file} format is in RGBA, not in RGB')
            continue

         self.setImageSize(x_data)

         return x_data, y_data, x_file, F

      return np.array([None]), np.array([None]), '', ''


   def shuffleImages(self):
      random.shuffle(self._imageList)

   def setImageSize(self, data):
      self.imageSize[0] = data[0]
      self.imageSize[1] = data[1]

   def setupBoundary(self, **kwargs):
      if 'r_lo' in kwargs:
         self.r_lo = kwargs['r_lo']
      if 'r_hi' in kwargs:
         self.r_hi = kwargs['r_hi']

      if 'c_lo' in kwargs:
         self.c_lo = kwargs['c_lo']
      if 'c_hi' in kwargs:
         self.c_hi = kwargs['c_hi']




   def getMatFromRange(self):
      pass

   def getRandomMatFromRange(self):
      pass

   def getRandomMat(self):
      pass













# class DataLoader:
#    def __init__(self, path2LstFile, shuffle=False, criteria=None):
#       self.idx = 0
#       self.Files = open(path2LstFile, 'r').readlines()
#       self.Files = [ line for line in self.Files if '#' not in line ]

#       # only select some images has the same Focus
#       if bool(criteria):
#          self.criteria = criteria
#          self.TestImgs = [ f for f in self.Files if self.criteria in f]
#       else:
#          self.TestImgs = self.Files

#       self.TotalImgs = len(self.TestImgs)

#       if shuffle:
#          random.shuffle(self.TestImgs)

#    def restart (self, shuffle=False):
#       self.idx = 0
#       if shuffle:
#          random.shuffle(self.TestImgs)

#    def getNextImgData(self, debug=False):
#       # line = 'init'
#       while self.idx < len(self.TestImgs):
#          line = self.TestImgs[self.idx]
#          self.idx += 1

#          x_file, y_file, Sensity, F = [ i for i in line.split(' ') if i != '' ]

#          if debug:
#             print(f'x_file=[{x_file}]')
#             print(f'y_file=[{y_file}]')

#          x_data = rawpy.imread(x_file).postprocess(use_camera_wb=True)
#          y_data = rawpy.imread(y_file).postprocess(use_camera_wb=True)
#          if len(x_data[0][0]) != 3:
#             print(f'Error: image {x_file} format is in RGBA, not in RGB')
#             continue
#          if len(y_data[0][0]) != 3:
#             print(f'Error: image {y_file} format is in RGBA, not in RGB')
#             continue

#          return x_data, y_data, x_file, F

#       return np.array([None]), np.array([None]), '', ''


# class DataConverter:
#    def __init__(self, **kwargs):
#       self.X_data = np.array(kwargs['X_data'], dtype=np.float64) / 255
#       self.Y_data = np.array(kwargs['Y_data'], dtype=np.float64) / 255

#       self.mat_c = kwargs['w']
#       self.mat_r = kwargs['h']

#       # self.returnMat = kwargs['']

#       self.max_r_ptr = max(len(self.X_data), len(self.Y_data)) - self.mat_r 
#       self.max_c_ptr = max(len(self.X_data[0]), len(self.Y_data[0])) - self.mat_c

#       self.totalMats = (self.max_r_ptr+1)*(self.max_c_ptr+1)


#       self.max_r_ptr_int = (max(floor(len(self.X_data)/self.mat_r), floor(len(self.Y_data)/self.mat_r)) - 1) * self.mat_r
#       self.max_c_ptr_int = (max(floor(len(self.X_data[0])/self.mat_c), floor(len(self.Y_data[0])/self.mat_c)) - 1) * self.mat_c

#       self.r_ptr = 0
#       self.c_ptr = 0

#       self.MatIdx = 1
#       self.totalMatsInt = (self.max_r_ptr_int / self.mat_r +1)*(self.max_c_ptr_int / self.mat_c +1)

#    def getNextMatrix(self):
#       if (self.c_ptr > self.max_c_ptr):
#          self.c_ptr  = 0
#          self.r_ptr += 1
#       if (self.r_ptr <= self.max_r_ptr):
#          mat_x = self.X_data[self.r_ptr : self.r_ptr+self.mat_r , self.c_ptr : self.c_ptr+self.mat_c , :]
#          mat_y = self.Y_data[self.r_ptr : self.r_ptr+self.mat_r , self.c_ptr : self.c_ptr+self.mat_c , :]

#          if (mat_x.shape != mat_y.shape):
#             print (f'Error when getting next matrix, r_ptr={self.r_ptr}, c_ptr={self.c_ptr}'+\
#                    f', mat_c={self.mat_c}, mat_r={self.mat_r}, X_data.shape={self.X_data.shape}, Y_data.shape={self.Y_data.shape}')
#             return np.array([None]), np.array([None])

#          self.MatIdx = (self.r_ptr)*(self.max_c_ptr+1) + self.c_ptr+1

#          self.c_ptr  += 1

#          # print (f'End with next matrix, r_ptr={self.r_ptr}, c_ptr={self.c_ptr}'+\
#          #           f', mat_c={self.mat_c}, mat_r={self.mat_r}, X_data.shape={self.X_data.shape}, max_c={self.max_c_ptr}, max_r={self.max_r_ptr}')

#          return mat_x.reshape(self.mat_r*self.mat_c, 3, 1), mat_y.reshape(self.mat_r*self.mat_c, 3, 1)
#       else:
#          # print (f'End with next matrix, r_ptr={self.r_ptr}, c_ptr={self.c_ptr}'+\
#          #           f', mat_c={self.mat_c}, mat_r={self.mat_r}, X_data.shape={self.X_data.shape}, max_c={self.max_c_ptr}, max_r={self.max_r_ptr}')
#          return np.array([None]), np.array([None])

#    def convert2ImageFromPrediction(self, y_image, y_data):
#       r_ptr = self.r_ptr

#       c_ptr = self.c_ptr - self.mat_c

#       if r_ptr+self.mat_r <= self.max_r_ptr:
#          print (f'r_ptr={r_ptr}, self.r_ptr={self.r_ptr}')
#          print (f'c_ptr={c_ptr}, self.c_ptr={self.c_ptr}')
#          y_image[r_ptr : r_ptr+self.mat_r , c_ptr : c_ptr+self.mat_c , :] = y_data.reshape(self.mat_r, self.mat_c, 3)

#    def getNextDataWindow(self):
#       if (self.c_ptr > self.max_c_ptr_int):
#          self.c_ptr  = 0
#          self.r_ptr += self.mat_r
#       if (self.r_ptr <= self.max_r_ptr_int):
#          mat_x = self.X_data[self.r_ptr : self.r_ptr+self.mat_r , self.c_ptr : self.c_ptr+self.mat_c , :]
#          mat_y = self.Y_data[self.r_ptr : self.r_ptr+self.mat_r , self.c_ptr : self.c_ptr+self.mat_c , :]

#          if (mat_x.shape != mat_y.shape):
#             print (f'Error when getting next matrix, r_ptr={self.r_ptr}, c_ptr={self.c_ptr}'+\
#                    f', mat_c={self.mat_c}, mat_r={self.mat_r}, X_data.shape={self.X_data.shape}, Y_data.shape={self.Y_data.shape}')
#             return np.array([None]), np.array([None])

#          self.MatIdx = int((self.r_ptr/self.mat_r)*(self.max_c_ptr_int/self.mat_r+1) + self.c_ptr/self.mat_c+1)

#          self.c_ptr  += self.mat_c

#          # print (f'End with next matrix, r_ptr={self.r_ptr}, c_ptr={self.c_ptr}'+\
#          #           f', mat_c={self.mat_c}, mat_r={self.mat_r}, X_data.shape={self.X_data.shape}, max_c={self.max_c_ptr}, max_r={self.max_r_ptr}')

#          return mat_x.reshape(self.mat_r*self.mat_c, 3, 1), mat_y.reshape(self.mat_r*self.mat_c, 3, 1)
#       else:
#          # print (f'End with next matrix, r_ptr={self.r_ptr}, c_ptr={self.c_ptr}'+\
#          #           f', mat_c={self.mat_c}, mat_r={self.mat_r}, X_data.shape={self.X_data.shape}, max_c={self.max_c_ptr}, max_r={self.max_r_ptr}')
#          return np.array([None]), np.array([None])

#    ##################################################################################################
#    def setWindowRange(self, debug=False, **kwargs):
#       c_low  = kwargs['c_low']
#       c_high = kwargs['c_high']
#       r_low  = kwargs['r_low']
#       r_high = kwargs['r_high']

#       if c_low >= 0 and c_low <= c_high and c_high <= self.max_c_ptr_int and \
#          r_low >= 0 and r_low <= r_high and r_high <= self.max_r_ptr_int :
#          self.c_low  = c_low
#          self.c_high = c_high
#          self.r_low  = r_low
#          self.r_high = r_high
#          self.range_r_ptr = r_low
#          self.range_c_ptr = c_low - self.mat_c # initial

#          self.total_range_mat = ((c_high - c_low)/self.mat_c + 1) * ((r_high - r_low)/self.mat_r + 1)

#          if debug:
#             String = ''
#             for key in kwargs.keys():
#                String  += f'{key}={kwargs[key]}, '
#             String += f'max_r_ptr_int={self.max_r_ptr_int}, '
#             String += f'max_c_ptr_int={self.max_c_ptr_int}, '
#             print(String)

#       else:
#          String = 'Invalid Range input '
#          for key in kwargs.keys():
#             String  += f'{key}={kwargs[key]}, '
#          String += f'max_r_ptr_int={self.max_r_ptr_int}, '
#          String += f'max_c_ptr_int={self.max_c_ptr_int}, '
#          print (String)
#          os._exit(1)

#    def nextRangeIdx(self):
#       if (self.range_c_ptr > self.c_high):
#          self.range_c_ptr = self.c_low
#          self.range_r_ptr += self.mat_r
#       else:
#          self.range_c_ptr += self.mat_c
#       if self.range_r_ptr > self.r_high:
#          return False
#       else:
#          return True

#    def getWindowMatDataInRange(self):
#       if self.nextRangeIdx():
#          mat_x = self.X_data[self.range_r_ptr : self.range_r_ptr+self.mat_r , self.range_c_ptr : self.range_c_ptr+self.mat_c , :]
#          mat_y = self.Y_data[self.range_r_ptr : self.range_r_ptr+self.mat_r , self.range_c_ptr : self.range_c_ptr+self.mat_c , :]

#          return mat_x.reshape(self.mat_r*self.mat_c, 3, 1), mat_y.reshape(self.mat_r*self.mat_c, 3, 1)
#       else:
#          print (f'--------self.range_r_ptr={self.range_r_ptr}')
#          return np.array([None]), np.array([None])


#    def convertPartImage(self, y_image, y_data, debug=False):
#       range_r_ptr = self.range_r_ptr - self.r_low

#       if (self.range_c_ptr == self.c_low):
#          range_c_ptr = self.range_c_ptr - self.c_low
#       else:
#          range_c_ptr = self.range_c_ptr - self.c_low - self.mat_c

#       if range_r_ptr+self.mat_r <= self.r_high - self.r_low:
#          if debug:
#             print (f'range_r_ptr={range_r_ptr}, self.range_r_ptr={self.range_r_ptr}')
#             print (f'range_c_ptr={range_c_ptr}, self.range_c_ptr={self.range_c_ptr}')
#          y_image[range_r_ptr : range_r_ptr+self.mat_r , range_c_ptr : range_c_ptr+self.mat_c , :] = y_data.reshape(self.mat_r, self.mat_c, 3)

#    def getPartImageShape(self):
#       return self.r_high - self.r_low + self.mat_r , self.c_high - self.c_low + self.mat_c , 3


# def main():
#    Data = DataLoader("C:\\MyFolder\\MyData\\QNN\\QNN\\Sony\\Sony_train_list.txt")
#    data_0 = Data.getNextImgData()
#    Con = DataConverter(X_data=data_0[0], Y_data=data_0[1], w=3, h=3)
#    check = Con.getNextMatrix()
#    print(check[0])
#    print(check[0].reshape(3,3,3,1))


# if __name__ == '__main__':
#    # main()
#    a = np.array( [ [ [3, 7, 8],
#                      [0, 1, 2],
#                      [11,7, 33]],

#                    [ [21, 44, 6],
#                      [0, 5, 9],
#                      [4, 89, 100]  ]

#                          ], dtype=np.float64)
#    print (a)
#    print (a.reshape(6,3,1))


