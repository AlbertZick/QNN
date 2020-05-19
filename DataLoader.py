import numpy as np
import rawpy
import matplotlib.pyplot as plt
import os
import random


def viewImage(data, save=False):
   fig  = plt.figure()
   plot = fig.add_subplot(1, 2, 1)
   plot.imshow(data[0])
   plot.set_title('X image')

   plot = fig.add_subplot(1, 2, 2)
   plot.imshow(data[1])
   plot.set_title('Y image')
   if True:
      name, _ = os.path.splitext(data[2].split('/')[-1])
      name = name + '.png'
      fig.savefig(name )

   fig.show()
   plt.show()

class DataLoader:
   def __init__(self, path2LstFile, shuffle=False):
      self.idx = 0
      self.Files = open(path2LstFile, 'r').readlines()
      if shuffle:
         random.shuffle(self.Files)

   def getNextImgData(self):
      # line = 'init'
      while self.idx <= len(self.Files):
         line = self.Files[self.idx]
         self.idx += 1

         x_file, y_file, Sensity, F = line.split(' ')
         x_data = rawpy.imread(x_file).postprocess(use_camera_wb=True)
         y_data = rawpy.imread(y_file).postprocess(use_camera_wb=True)
         if len(x_data[0][0]) != 3:
            print(f'Error: image {x_file} format is in RGBA, not in RGB')
            continue
         if len(y_data[0][0]) != 3:
            print(f'Error: image {y_file} format is in RGBA, not in RGB')
            continue

         return x_data, y_data, x_file, F

      return None, None, '', ''


class DataConverter:
   def __init__(self, **kwargs):
      self.X_data = np.array(kwargs['X_data'], dtype=np.float64) / 255
      self.Y_data = np.array(kwargs['Y_data'], dtype=np.float64) / 255

      self.mat_c = kwargs['h']
      self.mat_r = kwargs['w']

      self.max_r_ptr = max(len(self.X_data), len(self.Y_data)) - self.mat_r + 1
      self.max_c_ptr = max(len(self.X_data[0]), len(self.Y_data[0])) - self.mat_c + 1

      self.r_ptr = 0
      self.c_ptr = 0

   def getNextMatrix(self):
      if (self.c_ptr > self.max_c_ptr):
         self.c_ptr  = 0
         self.r_ptr += 1
      if (self.r_ptr <= self.max_r_ptr):
         mat_x = self.X_data[self.r_ptr : self.r_ptr+self.mat_r , self.c_ptr : self.c_ptr+self.mat_c , :]
         mat_y = self.Y_data[self.r_ptr : self.r_ptr+self.mat_r , self.c_ptr : self.c_ptr+self.mat_c , :]
         self.c_ptr  += 1

         return mat_x.reshape(self.mat_r*self.mat_c, 3, 1), mat_y.reshape(self.mat_r*self.mat_c, 3, 1)
      else:
         return None, None



def main():
   Data = DataLoader("C:\\MyFolder\\MyData\\QNN\\QNN\\Sony\\Sony_train_list.txt")
   data_0 = Data.getNextImgData()
   Con = DataConverter(X_data=data_0[0], Y_data=data_0[1], w=3, h=3)
   check = Con.getNextMatrix()
   print(check[0])
   print(check[0].reshape(3,3,3,1))


if __name__ == '__main__':
   # main()
   a = np.array( [ [ [3, 7, 8],
                     [0, 1, 2],
                     [11,7, 33]],

                   [ [21, 44, 6],
                     [0, 5, 9],
                     [4, 89, 100]  ]

                         ], dtype=np.float64)
   print (a)
   print (a.reshape(6,3,1))
