from Q_operator import RotateMatrix, RotateMatrixPolar, sigmoid
import numpy as np
from enum import Enum

class Updater_enum(Enum):
   Adam, SGD = range(2)

class Updater:
   def __init__(self, UpdateType=Updater_enum.SGD, **kwargs):
      self.UpdateType = UpdateType
      if self.UpdateType == Updater_enum.SGD:
         self.r = kwargs['r']

   def update(self, X, DX):
      if self.UpdateType == Updater_enum.SGD:
         return X - DX * self.r

class ActiveFunct_enum(Enum):
   sigm, tanh = range(2)

class ActiveFunct:
   def __init__(self, Type=ActiveFunct_enum.sigm):
      self.Type = Type

   def calc(self, X):
      if self.Type == ActiveFunct_enum.sigm:
         return sigmoid(X)

   def grad(self, X):
      if self.Type == ActiveFunct_enum.sigm:
         return sigmoid(X) * (1 - sigmoid(X))


class LossFunc_enum(Enum):
   diff = 1

class LossFunc:
   diff = 1
   def __init__(self, Type=LossFunc_enum.diff):
      self.Type = Type


   def calc(self, Predict, Real):
      if self.Type ==  LossFunc_enum.diff:
         return 1/2 * (Predict - Real) * (Predict - Real)

   def diff(self, Predict, Real):
      if self.Type ==  LossFunc_enum.diff:
         return np.absolute(Predict - Real)

class HiddenLayer:
   nth_layer = 0
   def __init__(self, i_dim, o_dim, next_o_dim, useBias=True):
      HiddenLayer.nth_layer += 1
      self.nth_layer = HiddenLayer.nth_layer

      self.useBias = useBias

      self.o_dim = o_dim
      self.i_dim = i_dim

      # O x I
      # theta follows U [-pi/2 ; pi/2]
      self.theta = np.random.rand(self.o_dim, self.i_dim)
      self.theta = np.array(self.theta, dtype=np.float64)*np.pi - np.pi/2
      self.z     = np.random.rand(self.o_dim, self.i_dim)
      self.z     = np.array(self.z, dtype=np.float64)*2*np.sqrt(6)/np.sqrt(o_dim + next_o_dim) - np.sqrt(6)/np.sqrt(o_dim + next_o_dim)

      # O x 3 x 1
      self.B  = np.random.rand( self.o_dim, 3, 1)

      # O x 3 x 1
      # self.K = rotate(X)
      self.K = np.zeros((self.o_dim, 3, 1), dtype=np.float64)

      self.RotateMat = np.zeros((self.o_dim, self.i_dim, 3, 3), dtype=np.float64)

      # for bachpropagation
      self.clrDWeight()

   def WrtModel(self, name):
      File = open(f'{name}_L{self.nth_layer}.txt', 'w')
      if self.useBias:
         File.write('useBias=True\n')
      else:
         File.write('useBias=False\n')

      File.write(f'layer={self.nth_layer}\no_dim={self.o_dim}\ni_dim={self.i_dim}\n')
      File.write(f'theta=\n')
      for i in range(len(self.theta)):
         for j in range(len(self.theta[0])):
            File.write(f'{self.theta[i,j]} ')
         File.write('\n')

      File.write(f'z=\n')
      for i in range(len(self.z)):
         for j in range(len(self.z[0])):
            File.write(f'{self.z[i,j]} ')
         File.write('\n')

      if self.useBias:
         File.write(f'B=\n')
         for i in range(self.o_dim):
            File.write(f'[{self.B[i,0,0]}  {self.B[i,1,0]}  {self.B[i,2,0]}]\n')


   def LoadModelFromFile(self):
      pass


   def setWeight(self, **kwargs):
      self.theta = kwargs['theta']
      self.z     = kwargs['z']
      self.B     = kwargs['B']

   def clrDWeight(self):
      self.Dtheta = np.zeros((self.o_dim, self.i_dim), dtype=np.float64)
      self.Dz     = np.zeros((self.o_dim, self.i_dim), dtype=np.float64)
      if self.useBias:
         self.DB   = np.zeros((self.o_dim, 4, 1), dtype=np.float64)

   def getWeight(self):
      return self.theta, self.z, self.B

   def compile(self, Update_c, ActFunc=ActiveFunct_enum.sigm):
      self.Update_c = Update_c
      self.ActFunc  = ActiveFunct(ActFunc)

  # input x in form   I x 3 x 1
   def forward(self, X):
      for i_out in range(self.o_dim):
         for i_in in range(self.i_dim):
            self.RotateMat[i_out, i_in] = RotateMatrixPolar(self.theta[i_out, i_in])
            self.K[i_out] = self.K[i_out] + self.z[i_out, i_in] * ( self.RotateMat[i_out, i_in] @ X[i_in] )

      if self.useBias:
         self.Y = self.ActFunc.calc(self.K + self.B)
      else:
         self.Y = self.ActFunc.calc(self.K)

      return self.Y

   def backprop(self, DY, X):
      # clear all differential
      self.clrDWeight()

      # DActFunc = DY * self.ActFunc.calc(self.K) * ( 1 - self.ActFunc.calc(self.K))
      if self.useBias:
         DActFunc = DY * self.ActFunc.grad(self.K + self.B)
      else:
         DActFunc = DY * self.ActFunc.grad(self.K)

      # print (f'DActFunc={DActFunc}')

      # We use average of all dimensions ijk
      for i_in in range(self.i_dim):
         for i_out in range(self.o_dim):

            ## Test code ################################################################################################################
            if self.nth_layer == 2:
               pass
               # print(f'calc={self.DerivativeMat(self.theta[i_out, i_in]) @ X[i_in]}')
               # print(f'z={self.z[i_out, i_in]}')
               # print(f'DActFuncDActFunc[i_out]={DActFunc[i_out]}')
            ##############################################################################################################################

            self.Dtheta[i_out, i_in] =\
                     np.average( DActFunc[i_out] * self.z[i_out, i_in] * (self.DerivativeMat(self.theta[i_out, i_in]) @ X[i_in]) )




      for i_in in range(self.i_dim):
         for i_out in range(self.o_dim):
            self.Dz[i_out, i_in] =\
                  np.average( DActFunc[i_out] * ( self.RotateMat[i_out, i_in] @ X[i_in] ) )

      if self.useBias:
         self.DB = DActFunc

      DX = np.zeros((self.i_dim, 3, 1))
      for i_in in range(self.i_dim):
         for i_out in range(self.o_dim):
            tmp_0    =  DActFunc[i_out] * self.z[i_out, i_in] * self.RotateMat[i_out, i_in]
            tmp_1    =  np.average( tmp_0 , axis=0)
            DX[i_in] =  DX[i_in] + tmp_1.reshape(3,1)

      return DX

   def update (self):
      self.theta = self.Update_c.update(self.theta, self.Dtheta)
      self.z     = self.Update_c.update(self.z, self.Dz)
      if self.useBias:
         self.B  = self.Update_c.update(self.B, self.DB)

   '''Additional functions'''
   def DerivativeMat(self, theta):
      df1 = -2/3 * np.sin(theta)
      df2 =  2/3 * np.sin(theta - np.pi/3)
      df3 =  2/3 * np.sin(theta + np.pi/3)

      DR = np.array( [  [df1, df2, df3],
                        [df3, df1, df2],
                        [df2, df3, df1] ], dtype=np.float64)

      return DR





# def main():
#    up = Updater(Updater_enum.SGD, r=0.001)
#    H1 = HiddenLayer(9, 5, 1, useBias=True)
#    H2 = HiddenLayer(5, 1, 1, useBias=True)

#    H1.compile(Update_c=up, ActFunc=ActiveFunct_enum.sigm)
#    H2.compile(Update_c=up, ActFunc=ActiveFunct_enum.sigm)

#    Loss = LossFunc()

#    # print ('== H1 ===========')
#    # print (H1.theta)
#    # print (H1.z)
#    # print (H1.B)
#    # theta_11, z_11, b_11 = H1.theta, H1.z, H1.B
#    # print ('== H2 ===========')
#    # print (H2.theta)
#    # print (H2.z)
#    # print (H2.B)
#    # theta_21, z_21, b_21 = H2.theta, H2.z, H2.B

#    x_data = np.random.rand(9,3,1)
#    y_data = np.random.rand(1,3,1)

#    H1_y = H1.forward(x_data)
#    H2_y = H2.forward(H1_y)

#    err  = Loss.calc(H2_y, y_data)

#    loss = Loss.diff(H2_y, y_data)
#    DH2_x = H2.backprop(loss, H1_y)
#    DH1_x = H1.backprop(DH2_x, x_data)

   # H1.update()
   # H2.update()

   # print ('== H1 ===========')
   # theta_12, z_12, b_12 = H1.theta, H1.z, H1.B
   # print (theta_12 - theta_11)
   # print (z_12 - z_11)
   # print (b_12 - b_11)
   # print ('== H2 ===========')
   # theta_22, z_22, b_22 = H2.theta, H2.z, H2.B
   # print (theta_22 - theta_21)
   # print (z_22 - z_21)
   # print (b_22 - b_21)

   # print(H2.Dz)
   # print(H2.z - 0.001 * H2.Dz)

   # H2.update()

   # print (H2.z)



   # print(H2.Dz)
   # print(H2.DB)


# if __name__ == '__main__':
#    main()