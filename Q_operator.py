import numpy as np
import math


def sigmoid(x):
  return 1/(1 + np.exp(-x))


def norm_Q(q):
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
  rst = np.zeros((4, 1), dtype=np.float64)

  rst[0,0] = a[0,0]*b[0,0] - a[1,0]*b[1,0] - a[2,0]*b[2,0] - a[3,0]*b[3,0]
  rst[1,0] = a[1,0]*b[0,0] + a[0,0]*b[1,0] - a[3,0]*b[2,0] + a[2,0]*b[3,0]
  rst[2,0] = a[2,0]*b[0,0] + a[3,0]*b[1,0] + a[0,0]*b[2,0] - a[1,0]*b[3,0]
  rst[3,0] = a[3,0]*b[0,0] - a[2,0]*b[1,0] + a[1,0]*b[2,0] + a[0,0]*b[3,0]

  return rst

def RotateMatrix(x):
  # https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
  s = 1 / ( norm_Q(x)*norm_Q(x) )
  R = np.zeros((3,3), dtype=np.float64)
  r, i, j, k = 0, 1, 2, 3

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

def RotateMatrixPolar (theta):
  R = np.zeros((3,3), dtype=np.float64)
  f1 = 1/3 + 2/3 * np.cos(theta)
  f2 = 1/3 - 2/3 * np.cos(theta - np.pi/3)
  f3 = 1/3 - 2/3 * np.cos(theta + np.pi/3)
  R[0,0] = f1
  R[0,1] = f2
  R[0,2] = f3

  R[1,0] = f3
  R[1,1] = f1
  R[1,2] = f2

  R[2,0] = f2
  R[2,1] = f3
  R[2,2] = f1

  return R


