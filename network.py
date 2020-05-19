from Q_operation import RotateMatrix

class Updater:
  Adam, SGD = range(2)

  def __init__(self, UpdateType=Updater.SGD, **kwargs):
    self.UpdateType = UpdateType

    if self.UpdateType == Updater.SGD:
      self.r = kwargs['r']

  def update(self, X, DX):
    if self.UpdateType == Updater.SGD:
      return X - DX * r

class ActiveFunct:
  sigm, tanh = range(2)

  def __init__(self, Type=ActiveFunct.sigm):
    self.Type = Type

  def calc(self, X):
    if self.Type = ActiveFunct.sigm:
      return sigmoid(X)

  def grad(self, X):
    if self.Type = ActiveFunct.sigm:
      return sigmoid(X) * (1 - sigmoid(X))


'''
class HiddenLayer:
  def __init__(self, i_dim, o_dim):   
    self.arg = arg

    self.o_dim = o_dim
    self.i_dim = i_dim

    # O x I x 4 x 1
    self.W  = np.random.rand(self.o_dim, self.i_dim, 4, 1)

    # O x 4 x 1
    self.B  = np.random.rand( self.o_dim, 4, 1)

    # O x 4 x 1
    # self.K = rotate(X)
    self.K = np.zeros((self.o_dim, 4, 1), dtype=np.float64)


  """
  Receive input x =  [ x1  x2 ... xn ] ^ T | x1,x2...xn are pure quaternion
  Return output y = sigmoid (K)
  """
  def forward(self, X):
    # self.K = Rotate(self.W) @ X
    for i_0 in range(self.o_dim):
      for i_1 in range(self.i_dim):
        self.K[i_0] += (Rotate(self.W[i_0, i_1]) @ X[i_1] ) / abs_Q(self.W[i_0, i_1])

    self.Y = sigmoid(self.K) + self.B


  # def backpro(self, DY, X):
  #   self.Dsigm = DY * sigmoid(self.K) * ( 1 - sigmoid(self.K))

  #   I = np.array( [[0], [1], [0], [0]] , dtype=np.float64 )
  #   J = np.array( [[0], [0], [1], [0]] , dtype=np.float64 )
  #   K = np.array( [[0], [0], [0], [1]] , dtype=np.float64 )


  #   for i_0 in range(self.o_dim):
  #     for i_1 in range(self.i_dim):
  #       self.DW[i_0, i_1] =  \
  #               self.Dsigm[i_0] *\
  #                 ( 
  #                  -conj_Q(mul_Q(self.W[i_0, i_1], X[i_0, i_1]))           + mul_Q(self.W[i_0, i_1], X[i_0, i_1]) +\
  #                   conj_Q(mul_Q(mul_Q(self.W[i_0, i_1], X[i_0, i_1]), I)) - mul_Q(mul_Q(self.W[i_0, i_1], X[i_0, i_1]), I) +\
  #                   conj_Q(mul_Q(mul_Q(self.W[i_0, i_1], X[i_0, i_1]), J)) - mul_Q(mul_Q(self.W[i_0, i_1], X[i_0, i_1]), J) +\
  #                   conj_Q(mul_Q(mul_Q(self.W[i_0, i_1], X[i_0, i_1]), K)) - mul_Q(mul_Q(self.W[i_0, i_1], X[i_0, i_1]), K) +\
  #                 ) / 4 / abs_Q(self.W[i_0, i_1])   +\

  #   self.DX = 00000000000000
  #   self.DB = self.Dsigm
'''

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

  def compile(self, Update_c=_c, ActFunc=ActiveFunct.sigm):
    self.Update_c = _c
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
    df2 =  2/3 * np.sin(theta + np.pi/3)

    DR = np.array( [ [df1, df2, df3],
                     [df3, df1, df2],
                     [df2, df3, df1] ], dtype=np.float64)
    
    return DR