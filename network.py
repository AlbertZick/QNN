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
   def __init__(self, i_dim, o_dim, useBias=True):
      HiddenLayer.nth_layer += 1
      self.nth_layer = HiddenLayer.nth_layer

      self.useBias = useBias

      self.o_dim = o_dim
      self.i_dim = i_dim

      # O x I
      self.theta = np.random.rand(self.o_dim, self.i_dim)
      self.z     = np.random.rand(self.o_dim, self.i_dim)

      # O x 3 x 1
      self.B  = np.random.rand( self.o_dim, 3, 1)

      # O x 3 x 1
      # self.K = rotate(X)
      self.K = np.zeros((self.o_dim, 3, 1), dtype=np.float64)

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
            self.K[i_out] = self.K[i_out] + self.z[i_out, i_in] * RotateMatrixPolar(self.theta[i_out, i_in]) @ X[i_in]


      if self.useBias:
         self.Y = self.ActFunc.calc(self.K + self.B)
      else:
         self.Y = self.ActFunc.calc(self.K)

      return self.Y

   def backprop(self, DY, X):
      # self.Dsigm = DY * self.ActFunc.calc(self.K) * ( 1 - self.ActFunc.calc(self.K))
      self.Dsigm = DY * self.ActFunc.grad(self.K)

      # We use average of all dimensions ijk
      for i_in in range(self.i_dim):
         for i_out in range(self.o_dim):
            self.Dtheta[i_out, i_in] =\
                     np.average( self.Dsigm[i_out] *\
                           self.z[i_out, i_in] * (self.DerivativeMat(self.theta[i_out, i_in]) @ X[i_in]) )

      for i_in in range(self.i_dim):
         for i_out in range(self.o_dim):
            self.Dz[i_out, i_in] =\
                  np.average( self.Dsigm[i_out] *\
                            RotateMatrixPolar(self.theta[i_out, i_in]) @ X[i_in] )

      if self.useBias:
         self.DB = self.Dsigm

      DX = np.zeros((self.i_dim, 3, 1))
      for i_in in range(self.i_dim):
         for i_out in range(self.o_dim):
            tmp_0    =  self.Dsigm[i_out] *\
                        self.z[i_out, i_in] * self.DerivativeMat(self.theta[i_out, i_in])
            tmp_1    =  np.average( tmp_0 , axis=0)
            DX[i_in] = DX[i_in] + np.reshape( tmp_1 , (3,1))

      return DX

   def update (self):
      self.Update_c.update(self.theta, self.Dtheta)
      self.Update_c.update(self.z, self.Dz)
      if self.useBias:
         self.Update_c.update(self.B, self.DB)

      # clear all differential
      self.clrDWeight()


   '''Additional functions'''
   def DerivativeMat(self, theta):
      df1 = -2/3 * np.sin(theta)
      df2 =  2/3 * np.sin(theta - np.pi/3)
      df3 =  2/3 * np.sin(theta + np.pi/3)

      DR = np.array( [  [df1, df2, df3],
                        [df3, df1, df2],
                        [df2, df3, df1] ], dtype=np.float64)

      return DR


