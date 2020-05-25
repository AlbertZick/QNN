import numpy as np
import rawpy
import matplotlib.pyplot as plt
import os
import random

class DataLoader:
   def __init__(self, path2LstFile, shuffle=False, criteria=None):
      self.idx = 0
      self.Files = open(path2LstFile, 'r').readlines()
      self.Files = [ line for line in self.Files if '#' not in line ]

      # only select some images has the same Focus
      if bool(criteria):
         self.criteria = criteria
         self.TestImgs = [ f for f in self.Files if self.criteria in f]
      else:
         self.TestImgs = self.Files

      self.TotalImgs = len(self.TestImgs)

      if shuffle:
         random.shuffle(self.TestImgs)

   def restart (self, shuffle=False):
      self.idx = 0
      if shuffle:
         random.shuffle(self.TestImgs)

   def getNextImgData(self):
      # line = 'init'
      while self.idx < len(self.TestImgs):
         line = self.TestImgs[self.idx]
         self.idx += 1

         x_file, y_file, Sensity, F = [ i for i in line.split(' ') if i != '' ]
         x_data = rawpy.imread(x_file).postprocess(use_camera_wb=True)
         y_data = rawpy.imread(y_file).postprocess(use_camera_wb=True)
         if len(x_data[0][0]) != 3:
            print(f'Error: image {x_file} format is in RGBA, not in RGB')
            continue
         if len(y_data[0][0]) != 3:
            print(f'Error: image {y_file} format is in RGBA, not in RGB')
            continue

         return x_data, y_data, x_file, F

      return np.array([None]), np.array([None]), '', ''


class DataConverter:
   def __init__(self, **kwargs):
      self.X_data = np.array(kwargs['X_data'], dtype=np.float64) / 255
      self.Y_data = np.array(kwargs['Y_data'], dtype=np.float64) / 255

      self.mat_c = kwargs['h']
      self.mat_r = kwargs['w']

      self.max_r_ptr = max(len(self.X_data), len(self.Y_data)) - self.mat_r 
      self.max_c_ptr = max(len(self.X_data[0]), len(self.Y_data[0])) - self.mat_c 

      self.r_ptr = 0
      self.c_ptr = 0

      self.MatIdx = 1
      self.totalMats = (self.max_r_ptr+1)*(self.max_c_ptr+1)

   def getNextMatrix(self):
      if (self.c_ptr > self.max_c_ptr):
         self.c_ptr  = 0
         self.r_ptr += 1
      if (self.r_ptr <= self.max_r_ptr):
         mat_x = self.X_data[self.r_ptr : self.r_ptr+self.mat_r , self.c_ptr : self.c_ptr+self.mat_c , :]
         mat_y = self.Y_data[self.r_ptr : self.r_ptr+self.mat_r , self.c_ptr : self.c_ptr+self.mat_c , :]

         if (mat_x.shape != mat_y.shape):
            print (f'Error when getting next matrix, r_ptr={self.r_ptr}, c_ptr={self.c_ptr}'+\
                   f', mat_c={self.mat_c}, mat_r={self.mat_r}, X_data.shape={self.X_data.shape}, Y_data.shape={self.Y_data.shape}')
            return np.array([None]), np.array([None])

         self.MatIdx = (self.r_ptr)*(self.max_c_ptr+1) + self.c_ptr+1

         self.c_ptr  += 1

         # print (f'End with next matrix, r_ptr={self.r_ptr}, c_ptr={self.c_ptr}'+\
         #           f', mat_c={self.mat_c}, mat_r={self.mat_r}, X_data.shape={self.X_data.shape}, max_c={self.max_c_ptr}, max_r={self.max_r_ptr}')

         return mat_x.reshape(self.mat_r*self.mat_c, 3, 1), mat_y.reshape(self.mat_r*self.mat_c, 3, 1)
      else:
         # print (f'End with next matrix, r_ptr={self.r_ptr}, c_ptr={self.c_ptr}'+\
         #           f', mat_c={self.mat_c}, mat_r={self.mat_r}, X_data.shape={self.X_data.shape}, max_c={self.max_c_ptr}, max_r={self.max_r_ptr}')
         return np.array([None]), np.array([None])

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
