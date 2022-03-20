import numpy as np
X = np.load('np444_15.npy', mmap_mode='r')
st ="""x...
....
o.x.
o..x"""
I = X["key"]==st
print(X[I])
