from Q_operator import RotateMatrix, RotateMatrixPolar, sigmoid, norm_Q
import numpy as np
from enum import Enum
import json



class Updater_enum(Enum):
   Adam, SGD = range(2)

class Updater:
   def __init__(self, UpdateType=Updater_enum.SGD, **kwargs):
      self.UpdateType = UpdateType
      if self.UpdateType == Updater_enum.SGD:
         self.r = kwargs['r']

   def update(self, X, DX):
      if self.UpdateType == Updater_enum.SGD:
         return X -  self.r * DX

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
         return Predict - Real

class NeuralCell:
   cell_id = 0

   def __init__(self):
      NeuralCell.cell_id += 1
      self.cell_id  = NeuralCell.cell_id
      self.type     = 'Unknown'
      self.nextCell = -1
      self.preCell  = -1

   def crtModel(modelDict):
      if modelDict['type'] == 'trigonometric_QCNN':
         model = trigonometric_QCNN(1,1,1,useBias=True)

      if modelDict['type'] == 'algebric_QCNN':
         model = algebric_QCNN(1,1,useBias=True)
         
      model.loadModel(modelDict)
      return model

   def model2Dict(self, RsltDict=None):
      if RsltDict == None:
         RsltDict = {}

      RsltDict['cell_id']  = self.cell_id
      RsltDict['nextCell'] = self.nextCell
      RsltDict['preCell']  = self.preCell
      return RsltDict


   def loadModel(self, modelDict):
      self.cell_id  = modelDict['cell_id']
      self.type     = modelDict['type']
      self.nextCell = modelDict['nextCell']
      self.preCell  = modelDict['preCell']


   def connect(self, cell):
      self.nextModel = cell.cell_id
      cell.preModel  = self.cell_id


class trigonometric_QCNN (NeuralCell):
   def __init__(self, i_dim, o_dim, next_o_dim, useBias=True):
      super().__init__()
      self.type = 'trigonometric_QCNN'

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

      # init Output
      self.clrOutput()

      # for bachpropagation
      self.clrDWeight()

   def WrtModel(self, name):
      File = open(f'{name}_L{self.cell_id}.txt', 'w')
      if self.useBias:
         File.write('useBias=True\n')
      else:
         File.write('useBias=False\n')

      File.write(f'layer={self.cell_id}\no_dim={self.o_dim}\ni_dim={self.i_dim}\n')
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


   def model2Dict(self, RsltDict=None):
      if RslDict == None:
         RsltDict = super().model2Dict()

      RsltDict['type']    = self.type
      RsltDict['useBias'] = self.useBias
      RsltDict['o_dim']   = self.o_dim
      RsltDict['i_dim']   = self.i_dim
      RsltDict['theta']   = self.theta.tolist()
      RsltDict['z']       = self.z.tolist()
      RsltDict['B']       = self.B.tolist()
      return RsltDict

   def loadModel(self, modelDict):
      super().loadModel(modelDict)
      self.type    = modelDict['type']
      self.useBias = modelDict['useBias']
      self.o_dim   = modelDict['o_dim']
      self.i_dim   = modelDict['i_dim']
      self.theta   = np.array(modelDict['theta'], dtype=np.float64)
      self.z       = np.array(modelDict['z'], dtype=np.float64)
      self.B       = np.array(modelDict['B'], dtype=np.float64)

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

   def clrOutput(self):
      # O x 3 x 1
      # self.K = rotate(X)
      self.K = np.zeros((self.o_dim, 3, 1), dtype=np.float64)
      self.Y = np.zeros((self.o_dim, 3, 1), dtype=np.float64)

      self.RotateMat = np.zeros((self.o_dim, self.i_dim, 3, 3), dtype=np.float64)

   def getWeight(self):
      return self.theta, self.z, self.B

   def compile(self, Update_c, ActFunc=ActiveFunct_enum.sigm):
      self.Update_c = Update_c
      self.ActFunc  = ActiveFunct(ActFunc)

  # input x in form   I x 3 x 1
   def forward(self, X):
      self.clrOutput()

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

      # We use average of all dimensions ijk
      for i_in in range(self.i_dim):
         for i_out in range(self.o_dim):

            ## Test code ################################################################################################################
            if self.cell_id == 2:
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


class algebric_QCNN(NeuralCell):
   def __init__(self, i_dim, o_dim, useBias=True):
      super().__init__()
      # trigonometric_QCNN.cell_id += 1
      # self.cell_id = trigonometric_QCNN.cell_id
      self.type = 'algebric_QCNN'

      self.useBias = useBias

      self.o_dim = o_dim
      self.i_dim = i_dim

      # O x I x 4 x 1
      self.W  = np.random.rand(self.o_dim, self.i_dim, 4, 1)

      # O x 3 x 1
      self.B  = np.random.rand( self.o_dim, 3, 1)

      self.clrOutPut()

      # for bachpropagation
      self.clrDWeight()

   def setWeight(self, **kwargs):
      self.W  = kwargs['W']
      self.B  = kwargs['B']

   def model2Dict(self, RsltDict=None):
      if RslDict == None:
         RsltDict = super().model2Dict()

      RsltDict['type']    = self.type
      RsltDict['useBias'] = self.useBias
      RsltDict['o_dim']   = self.o_dim
      RsltDict['i_dim']   = self.i_dim
      RsltDict['W']       = self.W.tolist()
      RsltDict['B']       = self.B.tolist()
      return RsltDict

   def loadModel(self, modelDict):
      super().loadModel(modelDict)
      self.type    = modelDict['type']
      self.useBias = modelDict['useBias']
      self.o_dim   = modelDict['o_dim']
      self.i_dim   = modelDict['i_dim']
      self.W       = np.array( modelDict['W'], dtype=np.float64)
      self.B       = np.array( modelDict['B'], dtype=np.float64)

   def compile(self, Update_c, ActFunc=ActiveFunct_enum.sigm):
      self.Update_c = Update_c
      self.ActFunc  = ActiveFunct(ActFunc)

   def clrDWeight(self):
      self.DW     = np.zeros((self.o_dim, self.i_dim, 4, 1), dtype=np.float64)
      if self.useBias:
         self.DB   = np.zeros((self.o_dim, 3, 1), dtype=np.float64)


   def clrOutPut(self):
      # O x 3 x 1
      # self.K = rotate(X)
      self.K = np.zeros((self.o_dim, 3, 1), dtype=np.float64)
      self.Y = np.zeros((self.o_dim, 3, 1), dtype=np.float64)

      self.RotateMat = np.zeros((self.o_dim, self.i_dim, 3,3), dtype=np.float64)
      self.norm      = np.zeros((self.o_dim, self.i_dim, 1), dtype=np.float64)


   def forward(self, X):
      self.clrOutPut()

      for i_out in range(self.o_dim):
         for i_in in range(self.i_dim):
            self.RotateMat[i_out, i_in] = RotateMatrix(self.W[i_out, i_in])
            self.norm[i_out, i_in]      = norm_Q(self.W[i_out, i_in])
            self.K[i_out] = self.K[i_out] + self.norm[i_out, i_in] * ( self.RotateMat[i_out, i_in] @ X[i_in] )

      if self.useBias:
         self.Y = self.ActFunc.calc(self.K + self.B)
      else:
         self.Y = self.ActFunc.calc(self.K)

      return self.Y

   def backprop(self, DY, X):
      # clear all differential
      self.clrDWeight()

      DX = np.zeros((self.i_dim, 3, 1), dtype=np.float64)

      # DActFunc = DY * self.ActFunc.calc(self.K) * ( 1 - self.ActFunc.calc(self.K))
      if self.useBias:
         DActFunc = DY * self.ActFunc.grad(self.K + self.B)
      else:
         DActFunc = DY * self.ActFunc.grad(self.K)

      if self.useBias:
         self.DB = DActFunc

      for i_out in range(self.o_dim):
         for i_in in range(self.i_dim):

            W_o_i   = np.reshape( self.W[i_out, i_in], (4,))
            # print (W_o_i)
            # print (W_o_i.shape)
            # a,b,c,d = W_o_i.reshape(1,4)

            a,b,c,d = np.reshape( self.W[i_out, i_in], (4,) )
            x,y,z   = np.reshape( X[i_in], (3,) )

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

            self.DW[i_out,i_in,0,0] = (   DActFunc[i_out,0,0] * dR_da +\
                                          DActFunc[i_out,1,0] * dG_da +\
                                          DActFunc[i_out,2,0] * dB_da )/3
            self.DW[i_out,i_in,1,0] = (   DActFunc[i_out,0,0] * dR_db +\
                                          DActFunc[i_out,1,0] * dG_db +\
                                          DActFunc[i_out,2,0] * dB_db )/3
            self.DW[i_out,i_in,2,0] = (   DActFunc[i_out,0,0] * dR_dc +\
                                          DActFunc[i_out,1,0] * dG_dc +\
                                          DActFunc[i_out,2,0] * dB_dc )/3
            self.DW[i_out,i_in,3,0] = (   DActFunc[i_out,0,0] * dR_dd +\
                                          DActFunc[i_out,1,0] * dG_dd +\
                                          DActFunc[i_out,2,0] * dB_dd )/3

            tmp_0 = np.transpose(DActFunc[i_out] * self.RotateMat[i_out, i_in])
            tmp_1 = np.average(tmp_0 , axis=1).reshape(3,1)
            DX[i_in] += tmp_1

      DX = DX / self.o_dim

      return DX

   def update (self):
      self.W = self.Update_c.update(self.W, self.DW)
      if self.useBias:
         self.B  = self.Update_c.update(self.B, self.DB)



##############################################################################################################################
def main():
   up = Updater(Updater_enum.SGD, r=0.01)
   # H1 = algebric_QCNN(9, 1, 1, useBias=True)
   H1 = trigonometric_QCNN(9, 1, 1, useBias=True)

   H1.compile(Update_c=up, ActFunc=ActiveFunct_enum.sigm)

   Loss = LossFunc()

   json_ob = json.dumps(H1.model2Dict())

   print (json_ob)

   File = open('model.txt', 'w')
   File.write(json_ob)
   File.close()

   File = open('model.txt', 'r')
   data = json.load(File)

   # iteration = 0

   # x_data = np.random.rand(9,3,1)
   # y_data = np.random.rand(1,3,1)

   # max_i = 100000

   # while iteration<max_i:


   #    H1_y = H1.forward(x_data)

   #    # print (f'H1_y={H1_y}')
   #    # print (f'y_data={y_data}')

   #    err  = Loss.calc(H1_y, y_data)

   #    loss = Loss.diff(H1_y, y_data)
   #    # print (f'loss ={loss}')
   #    DH1_x = H1.backprop(loss, x_data)

   #    H1.update()

   #    if iteration in range(0, max_i, 10):
   #       print (f'iter={iteration}, err={err}')
   #    # print (f'H1.DB = {H1.DB}')
   #    # print (f'H1.K = {H1.K}')
   #    iteration += 1

   # print(f'H1_theta={H1.theta}')


if __name__ == '__main__':
   main()