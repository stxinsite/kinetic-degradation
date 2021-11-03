import numpy as np
import pandas as pd

# this function is like rates_ternary_formation()
def f(x, TT, *y):
    # this function is like dTernary_Ubdt()
    def g(arr):
        return arr[0] + arr[1]

    print(x, TT, y)
    if len(y):  # if y has non-zero length
        y_arr = np.insert(np.array(y), 0, TT)
        y_pairs = np.lib.stride_tricks.sliding_window_view(y_arr, 2)
        g_arr = np.apply_along_axis(g, 1, y_pairs).tolist()
        print(y_pairs)
    else:
        g_arr = np.empty(0)
    print(g_arr)
    print(type(g_arr))

z = np.array([0, 1, 2, 3, 4, 5] + [6, 7])
f(*z)

np.sum((1,2,3))
np.sum(())
q = np.array([0, 1,])
f(*q)
len(())
a = np.array(
    [1, 2, 3]
    + [0] * 3
)  #
a
b = np.empty(0)

b = np.empty((0, 5))
b

np.append(b, [[1,2,3,4,5], [6, 7, 8, 9, 19]], axis = 0)

(1,2,3,)[-1]



np.concatenate((a,b))

def h(x, *y):
    def r(p, *q):
        print(q)
        print(len(q))
    print(len(y))
    return r(x, *y)

arr = np.array([1, 2, 3, 4])
arr1 = np.array([1])

s = (
    [1,2,3]
    + [4]
)
type(s)

l = []
l.append([1,2,3])
l.append([4,5,6])

l

o = [[7,8,9], [10, 11, 12]]
o

l + o
np.array(l+o)
np.array(
    [
        [1,2,3],
        [4,5,6]
    ]
)

['Ternary_Ub_' + str(i) for i in range(1, 0 + 1)]

h(*arr)
h(*arr1)

[1] * -1

some_condition = False

5.4 * (2 if some_condition else 0)

[1] * (1 if some_condition else 0)

qu = np.array([[1,2,3], [4,5,6]])
qi = np.array([[7,8,9]])
np.concatenate((qu, qi))
