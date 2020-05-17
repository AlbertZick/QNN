import numpy as np
import math


def sigmoid(x):
  return 1/(1 + np.exp(-x))


def abs_Q(q):
  return math.sqrt( q[0,0]*q[0,0] + q[1,0]*q[1,0] + q[2,0]*q[2,0] + q[3,0]*q[3,0] )


def conj_Q(q):
  rst = np.zeros((4,1), dtype=np.float64)
  rst[0,0] =  q[0,0]
  rst[1,0] = -q[1,0]
  rst[2,0] = -q[2,0]
  rst[3,0] = -q[3,0]
  return rst

def Q2M(x):
  rst = np.zeros((2,2), dtype=np.float64)

  rst[0,0] =  x[0,0] + x[1,0]*j
  rst[0,1] =  x[2,0] + x[3,0]*j
  rst[1,0] = -x[2,0] + x[3,0]*j
  rst[1,1] =  x[0,0] - x[1,0]*j

  return rst

def M2Q(x):
  rst = np.zeros((4,1), dtype=np.float64)

  if (np.real(x[0,0]) != np.real(x[1,1])):
    print ("Error: 1")
  if (np.imag(x[0,0]) != -np.imag(x[1,1])):
    print ("Error: 2")
  if (np.real(x[0,1]) != -np.real(x[1,0])):
    print ("Error: 3")
  if (np.imag(x[0,1]) != np.imag(x[1,0])):
    print ("Error: 4")
  
  rst[0,0] = np.real(x[0,0])
  rst[1,0] = np.imag(x[0,0])
  rst[2,0] = np.real(x[0,1])
  rst[3,0] = np.imag(x[0,1])

  return rst


def mul_Q(a, b):
  result = np.zeros((4, 1), dtype=np.float64)

  result[0,0] = a[0,0]*b[0,0] - a[1,0]*b[1,0] - a[2,0]*b[2,0] - a[3,0]*b[3,0]
  result[1,0] = a[1,0]*b[0,0] + a[0,0]*b[1,0] - a[3,0]*b[2,0] + a[2,0]*b[3,0]
  result[2,0] = a[2,0]*b[0,0] + a[3,0]*b[1,0] + a[0,0]*b[2,0] - a[1,0]*b[3,0]
  result[3,0] = a[3,0]*b[0,0] - a[2,0]*b[1,0] + a[1,0]*b[2,0] + a[0,0]*b[3,0]

  return result

def RotateMatrix(x):
  # https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
  s = 1 / ( abs_Q(x)*abs_Q(x) )
  R = np.zeros((3,3), dtype=np.float64)
  r = 0
  i = 1
  j = 2
  k = 3

  R[0,0] =  1 - 2*s*( x[j,0]**2 + x[k,0]**2 )
  R[0,1] =  2*s*( x[i,0]*x[j,0] - x[k,0]*x[r,0] )
  R[0,2] =  2*s*( x[i,0]*x[k,0] + x[j,0]*x[r,0] )

  R[1,0] =  2*s*( x[i,0]*x[j,0] + x[k,0]*x[r,0] )
  R[1,1] =  1 - 2*s*( x[i,0]**2 + x[k,0]**2 )
  R[1,2] =  2*s*( x[j,0]*x[k,0] - x[i,0]*x[r,0] )

  R[2,0] =  2*s*( x[i,0]*x[k,0] - x[j,0]*x[r,0] )
  R[2,1] =  2*s*( x[j,0]*x[k,0] + x[i,0]*x[r,0] )
  R[2,2] =  1 - 2*s*( x[i,0]**2 + x[j,0]**2 )

  return R













































mat_A = np.array( [[1 + 2j, 3 + 4j],
                   [-3+ 4j, 1 - 2j]] )

mat_B = np.array( [[9 + 8j, 4 + 5j],
                   [-4+ 5j, 9 - 8j]] )


# a = M2Q(mat_A)
# b = M2Q(mat_B)
# print (a)
# print (b)

# print (M2Q(mat_A@mat_B) == mul_Q(a, b))
# print (mul_Q(a, b))


mat_a = np.array([[0],
                  [12],
                  [3],
                  [4]])

mat_w = np.array([[8],
                  [13],
                  [10],
                  [18]])

print ( mul_Q( mul_Q(mat_w, mat_a), conj_Q(mat_w) /abs_Q(mat_w) ) )

print (( RotateMatrix(mat_w) @ mat_a[1:] ) * abs_Q(mat_w) )
