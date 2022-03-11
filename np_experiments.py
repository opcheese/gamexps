import numpy as np

a = np.arange(27)
c = np.arange(9).reshape((3,3))
b = a.reshape([3,3,3])

# b = np.copy(a)
# b1 = b.reshape([3,3])
# print(b1)
for i in range(1,4):
    b2=np.rot90(b,i,(1,2))
    print(b2)
    l = np.rot90(c,i)
    print(np.reshape(l,-1))

    r = np.flip(b,2)
    print(r)
    g = np.flip(c,1)
    print(g)
# print(b)

# b = np.rot90(b,2,(1,2))
# print(b)

# c = np.arange(9).reshape((3,3))
# b=  np.flip(c,)
# print(b)