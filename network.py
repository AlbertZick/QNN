
from Q_operation import RotateMatrix


class Updater:
  def __init__(self, r):
    self.r = r

  def update(self, X, DX):
    return X - DX * r



class HiddenLayer:
  def __init__(self, i_dim, o_dim):   
    self.arg = arg

    self.o_dim = o_dim
    self.i_dim = i_dim

    # O x I x 3
    self.W  = np.random.rand(self.o_dim, self.i_dim, 4)
    # O x 1
    self.B  = np.random.rand( self.o_dim, 1)

    # self.K = rotate(X)


  """
  Receive input x =  [ x1  x2 ... xn ] ^ T | x1,x2...xn are pure quaternion
  Return output y = sigmoid (K)
  """
  def forward(self, x):
    self.K = np.zeros((self.o_dim, 3), dtype=np.float64)
    # self.K = Rotate(self.W) @ x
    for i_0 in range(self.o_dim):
      self.K[i_0] = Rotate(self.W[i_0]) @ x

    self.Y = sigmoid(self.K) + self.B


  def backpro(self, DY):
    self.DW = 
    self.DX = 
    self.DB = 



