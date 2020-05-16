import 








  self.K = np.zeros((self.o_dim, 3), dtype=np.float64)
  # self.K = Rotate(self.W) @ x

  for i_0 in range(len(self.o_dim)):
    self.K[i_0] = Rotate(self.W[i_0]) @ x

  self.Y = sigmoid(self.K) + self.B


def backpro(self, DY):
  self.DW = 
  self.DX = 
  self.DB = 

class Updater:
  def __init__(self, r):
    self.r = r

  def update(self, X, DX):
    return X - DX * r
