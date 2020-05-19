

from Q_operation import RotateMatrix


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
   def forward(x):
      self.K = np.zeros((self.o_dim, 3), dtype=np.float64)
      for i_0 in range(self.o_dim):
         self.K = RotateMatrix(self.W[i]) @ self.


      