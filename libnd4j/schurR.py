import scipy
import tensorflow as tf
import numpy as np
import math


def prodgiv(c,s,n):
    N = n - 1 
    p = N - 1
    q = N - 2

    Q = np.eye(n)
    Q[p,p] = c[p]
    Q[N,N] = c[p]
    Q[p,N] = s[p]
    Q[N,p] = - s[p]
    for k in range(q, -1, -1): # down to 0
        k1 = k + 1
        Q[k,k] = c[k]
        Q[k1, k] = -s[k]
        qq = Q[k1, k1:n]
        Q[k, k1:n] = s[k] * qq;
        Q[k1, k1:n] = c[k] * qq;
    # next k
    return Q

def givcos(a,b):
    c = 1.
    s = 0.
    if (b == 0.) :
        return (c,s)
    elif abs(b) > abs(a):
        t = -a / b
        s = 1. / math.sqrt(1 + t ** 2)
        c = s * t
    else:
        t = -b / a
        c = 1. / math.sqrt(1 + t ** 2)
        s = c * t
    return (c,s)
# end givcos

def garow(M, c, s, i, k, j1, j2):
    for j in range(j1,j2):
       t1 = M[i, j]
       t2 = M[k, j]
       M[i,j] = c*t1 - s * t2
       M[k,j] = s*t1 + c * t2
    #next j
    return M
# end garow

def gacol(M, c, s, i, k, j1, j2):
    for j in range(j1,j2):
       t1 = M[i, j]
       t2 = M[k, j]
       M[i,j] = c*t1 - s * t2
       M[k,j] = s*t1 + c * t2
    #next j
    return M
# end garow

def qrgivens(h):
    m = h.shape[-2]
    n = h.shape[-1]
    c = []
    s = []
    for k in range(n - 1):
        cScalar, sScalar = givcos(h[k,k], h[k+1,k])
        c.append(cScalar); s.append(sScalar)
        h = garow(h, c[k], s[k], k, k+1, k, n)
    # next k
    r = h
    q = prodgiv(c,s,n)
    return (q, r, c, s)
#end qrgivens

def hessqr(A, niter):
     n = A.shape[-1]
     t, Qhess = scipy.linalg.hessenberg(A, True) # reduce to hessenberg with generated transformation matrix
     for j in range(niter):
         Q, R, c, s = qrgivens(t)
         print(Q, R, c, s)
         t = R
         for k in range(n-1):
             t = gacol(t, c[k], s[k], 0, k + 1, k, k + 1) 
         # next k
     # next j
     return (t, Q, R)
#end hessqr

def householderReflector(x,y,z):
    print(x,y,z)
    u = tf.constant((x,y,z), shape=(3,1), dtype = tf.float64)
    u1 = u / tf.norm(u) # normalize 
    e1 = tf.constant((1., 0., 0.), dtype = tf.float64)
    e1 *= math.copysign(1, x)
    us = u1 - e1
    w = us / tf.norm(us)
    P = tf.eye(3, dtype=tf.float64) - 2. * tf.linalg.matmul(w, w, False, True)
    return P.numpy()
# householderReflector

def givens(x,y):
    c = x / math.sqrt(x**2 + y**2)
    s = y / math.sqrt(x**2 + y**2)
    G = tf.constant((c, -s, s, c), shape=(2,2), dtype=tf.float64)
    return G.numpy()
# givens

def Francis(H):
    n = H.shape[-1]
    p = n - 1
    hH = H.numpy()
    iter = 0
    eps = 1.e-6
    while p > 2:
        q = p - 1
        s = hH[q,q] + H[p,p] 
        t = hH[q,q] * hH[p,p]
        # compute first 3 elements of the first column of H
        x = (hH[0,0] ** 2 + hH[0,1] * hH[1,0] - s * hH[0,0] + t).numpy()
        y = (hH[1,0] * (hH[0,0] + hH[1,1] - s)).numpy()
        z = hH[1,0] * hH[2,1]
        # householder transformation for columns
        for k in range(q):
            P = householderReflector(x,y,z)
            r = min(p + 1, k + 3)
            R = min(p + 1, k + 4) 
            hH3P = hH[k:r, k:p + 1] # retrieve 3 rows from the k-th
            hH[k:k+3, k:p + 1] = tf.linalg.matmul(P, hH3P)
            hHP3 = hH[0:R, k:k+3]
            hH[0:R, k:k+3] = tf.linalg.matmul(hHP3, P)
            x = hH[k+1, k]; y = hH[k+2, k]
            if k < p - 3:
                z = hH[k+3, k]
        #next k
        # givens transformation
        G = givens(x,y)
        hH2P = hH[q:p+1, p-2:n] # 2x3?
        hH[q:p+1, p-2:n] = tf.linalg.matmul(G, hH2P, True, False)
        hHP2 = hH[0:p+1, p - 1:p+1] # Px2 
        hH[0:p+1, p - 1: p+ 1] = tf.linalg.matmul(hHP2, G)
        if abs(hH[p,q]) < eps * (abs(hH[q,q]) + abs(hH[p,p])):
             hH[p,q] = 0.; p -= 1
        elif abs(hH[p-1,q-1]) < eps * (abs(hH[q,q]) + abs(hH[q-1,q-1])):
             hH[p-1,q-1] = 0.; p -= 2
        else:
            iter += 1
            print("Convergence has not achieved yet. Continue with %d iteration." % iter)
            if (iter > 100):
               print("Result has not achieved with proper density. Breaking on %d iteration" % iter)
               break
        #endif
        return hH
    # end while             
if __name__ == '__main__':
     A = tf.constant((17,24,1, 8, 15, 23, 5, 7, 14, 16, 4, 6, 13, 20, 22, 10, 12, 19, 21, 3, 11, 18, 25, 2, 9), shape=(5,5), dtype=tf.float32)
     anA = A.numpy()

     t, Q, R = hessqr(anA, 60)
     print('T is \n', t); print('\nQ is \n', Q); print('R is \n', R)

     T, U = scipy.linalg.schur(A)
     print("T=\n", T)
     print("\nU=\n", U)

     print("==============================================================================================================")
     H = scipy.linalg.hessenberg(A)
     hH = tf.constant(H, dtype=tf.float64)
     tT = Francis(hH)
     print("Francis has", tT) 
