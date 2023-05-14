# dimesion of an array is known as rank.

import numpy as np


# # arr1 with rank one
# arr1=np.array([1,2,3,4])
# print(arr1)

# # arr1 with rank two using list
# arr2=np.array([[1,2,3,4],
#                [1,2,3,4]])
# print(arr2)

# # arr1 with rank one using tuples
# arr3=np.array((1,2,3,4))
# print(arr3)
 
# # Initial Array
# arr = np.array([[-1, 2, 0, 4],
#                 [4, -0.5, 6, 0],
#                 [2.6, 0, 7, 8],
#                 [3, -7, 4, 2.0]])
# print("Initial Array: ")
# print(arr)
 
# # Printing a range of Array
# # with the use of slicing method
# sliced_arr = arr[:2,::2]
# print ("Array with first 2 rows and alternate columns(0 and 2):\n", sliced_arr)
 
# Printing elements at
# specific Indices
# Index_arr = arr[[1, 1, 0, 3], 
#                 [3, 2, 1, 0]]
# print ("\nElements at indices (1, 3), (1, 2), (0, 1), (3, 0):\n", Index_arr)

# # Defining Array 1
# a = np.array([[1, 2],
#               [3, 4]])
 
# # Defining Array 2
# b = np.array([[4, 3],
#               [2, 1]])
               
# # Adding 1 to every element
# print ("Adding 1 to every element:\n", a + 1)
 
# # Subtracting 2 from each element
# print ("\nSubtracting 2 from each element:\n", b - 2)
 
# # sum of array elements
# # Performing Unary operations
# print ("\nSum of all array elements:\n", a.sum())
 
# # Adding two arrays
# # Performing Binary operations
# print ("\nArray sum:\n", a + b)

# # Integer datatype
# # guessed by Numpy
# x = np.array([1, 2])  
# print("Integer Datatype: ")
# print(x.dtype)         
 
# # Float datatype
# # guessed by Numpy
# x = np.array([1.0, 2.0]) 
# print("\nFloat Datatype: ")
# print(x.dtype)  
 
# # Forced Datatype
# x = np.array([1, 2], dtype = np.int64)   
# print("\nForcing a Datatype: ")
# print(x.dtype)

# # First Array
# arr1 = np.array([[4, 7], [2, 6]], 
#                  dtype = np.float64)
                  
# # Second Array
# arr2 = np.array([[3, 6], [2, 8]], 
#                  dtype = np.float64) 
 
# # Addition of two Arrays
# Sum = np.add(arr1, arr2)
# Sum = arr1+arr2
# print("Addition of Two Arrays: ")
# print(Sum)

 
# # Addition of all Array elements
# # using predefined sum method
# Sum1 = np.sum(arr1)
# # print(arr1.sum())
# print("\nAddition of Array elements: ")
# print(Sum1)
 
# # Square root of Array
# Sqrt = np.sqrt(arr1)
# print("\nSquare root of Array1 elements: ")
# print(Sqrt)
 
# # Transpose of Array
# # using In-built function 'T'
# Trans_arr = arr1.T
# print("\nTranspose of Array: ")
# print(Trans_arr)

# # Creating array object
# arr = np.array( [[ 1, 2, 3],
#                  [ 4, 2, 5]] )
 
# # Printing type of arr object
# print("Array is of type: ", type(arr))
 
# # Printing array dimensions (axes)
# print("No. of dimensions: ", arr.ndim)
 
# # Printing shape of array
# print("Shape of array: ", arr.shape)
 
# # Printing size (total number of elements) of array
# print("Size of array: ", arr.size)
 
# # Printing type of elements in array
# print("Array stores elements of type: ", arr.dtype)

# Creating array from list with type float
# a = np.array([[1, 2, 4], [5, 8, 7]], dtype = 'float')
# print ("Array created using passed list:\n", a)
 
# # Creating array from tuple
# b = np.array((1 , 3, 2))
# print ("\nArray created using passed tuple:\n", b)
 
# # Creating a 3X4 array with all zeros
# c = np.zeros((3, 4),dtype = 'int')
# print ("\nAn array initialized with all zeros:\n", c)
 
# # Create a constant value array of complex type
# d = np.full((3, 3), 6, dtype = 'complex')
# print ("\nAn array initialized with all 6s.Array type is complex:\n", d)


# # An exemplar array
# arr = np.array([[-1, 2, 0, 4],
#                 [4, -0.5, 6, 0],
#                 [2.6, 0, 7, 8],
#                 [3, -7, 4, 2.0]])
 
# # boolean array indexing example
# cond = arr > 0 # cond is a boolean array
# temp = arr[cond]
# print ("\nElements greater than 0:\n", temp)
# print(cond)

# arr = np.array([[1, 5, 6],
#                 [4, 7, 2],
#                 [3, 1, 9]])
 
# # maximum element of array
# print ("Largest element is:", arr.max())
# print ("Row-wise maximum elements:",arr.max(axis = 1))
 
# # minimum element of array
# print ("Column-wise minimum elements:",arr.min(axis = 0))
 
# # sum of array elements
# print ("Sum of all array elements:",arr.sum())
 
# # cumulative sum along each row
# print ("Cumulative sum along each row:\n",arr.cumsum(axis = 1))

# a = np.array([[1, 2],
#             [3, 4]])
# b = np.array([[4, 3],
#             [2, 1]])
 
# # add arrays
# print ("Array sum:\n", a + b)
 
# # multiply arrays (elementwise multiplication)
# print ("Array multiplication:\n", a*b)
 
# # matrix multiplication
# print ("Matrix multiplication:\n", a.dot(b))

# create an array of sine values
# a = np.array([0, np.pi/2, np.pi])
# print ("Sine values of array elements:\n", np.sin(a))
 
# # exponential values
# a = np.array([0, 1, 2, 3])
# print ("Exponent of array elements:\n", np.exp(a))
 
# # square root of array values
# print ("Square root of array elements:\n", np.sqrt(a))

# print(np.pi)

# np.int16 is converted into a data type object.
# print(np.dtype(np.int16))

# i4 represents integer of size 4 byte
# > represents big-endian byte ordering and
# < represents little-endian encoding.
# dt is a dtype object
# dt = np.dtype('>i4')
# print(dt)
# print(type(dt))
# print(dt.dtype)
 
# print("Byte order is:",dt.byteorder)
 
# print("Size is:",dt.itemsize)
 
# print("Data type is:",dt.name)


# Python Program illustrating
# numpy.all() method
 
# Parameter : keepdims     
# # setting keepdims = True
# print("\nBool Value : ", np.all([[1, 0],[0, 4]], True))
# # setting keepdims = True
# print("\nBool Value : ", np.all([[0, 0],[0, 0]], False))

# Python Program illustrating
# numpy.all() method

# Axis = NULL
# True False
# True True
# True : False = False

# print("Bool Value with axis = NONE : ",np.all([[True,False],[True,True]]))

# Axis = 0
# True False
# True True
# True : False
# print("\nBool Value with axis = 0 : ",np.all([[True,False],[True,True]], axis = 0))
# print("\nBool : ", np.all([3,1,2]))

# # Not a Number (NaN), positive infinity and negative infinity
# # evaluate to True because these are not equal to zero.
# # print("\nBool : ", np.all([1.0, np.nan]))
# print("\nBool Value : ", np.all([[0, 0],[0, 0]]))
# print("\nBool Value : ", np.all([[1, 0],[0, 4]]))
# # setting keepdims = True
# print("\nBool Value : ", np.all([[0, 0],[0, 0]]))


# Python Programming illustrating
# numpy.arange method
# print("A\n", np.arange(4).reshape(2, 2), "\n")
# print("A\n", np.arange(4, 10), "\n")
# print("A\n", np.arange(4, 20, 3), "\n")
# print(np.arange(1, 2, 0.1))

# Python Program illustrating
# numpy.dot() method


# Scalars
# product = np.dot(5, 4)
# print("Dot Product of scalar values : ", product)
# # 1D array
# vector_a = 2 + 3j
# vector_b = 4 + 5j
# product = np.dot(vector_a, vector_b)
# print("Dot Product : ", product)
'''
How Code1 works ? 
vector_a = 2 + 3j 
vector_b = 4 + 5j
now dot product = 2(4 + 5j) + 3j(4 +5j) = 8 + 10j + 12j - 15 = -7 + 22j
here vector_a and vector_b is complex numbers
'''

# a=np.random.rand(3,3)
# a=np.linspace(2,9,50)
# a=np.random.randn(3,3)
# a=np.random.randint(1,100,15).reshape(3,5)
# a=np.random.random_sample((1,5))
# print(a)


# A structured data type containing a
# 16-character string (in field ‘name’) 
# and a sub-array of two 64-bit floating
# -point number (in field ‘grades’)
 

# dt = np.dtype([('name', np.unicode_, 16),
#                ('grades', np.float64, (2,))])
# # Data type of object with field grades
# print(dt)
# print(dt['grades'])
# # Data type of object with field name 
# print(dt['name'])
# # x is a structured array with names
# # and marks of students.
# # Data type of name of the student is 
# # np.unicode_ and data type of marks is 
# # np.float(64)
# x = np.array([('Sarah', (8.0, 7.0)),
#               ('John', (6.0, 7.0))], dtype=dt)
# print(x)
# print(x[1])
# print("Grades of John are: ", x[1]['grades'])
# print("Names are: ", x['name'])

# # sort along the first axis
# a = np.array([[12, 15], 
#               [10, 1]])
# arr1 = np.sort(a, axis = 0)        
# print ("Along first axis : \n", arr1)        
# # sort along the last axis
# a = np.array([[10, 15],
#               [12, 1]])
# arr2 = np.sort(a, axis = 1)        
# print ("\nAlong first axis : \n", arr2)
# a = np.array([[12, 15], 
#               [10, 1]])
# arr1 = np.sort(a, axis = None)        
# print ("\nAlong none axis : \n", arr1.reshape(2,2))


# # Numpy array created
# a = np.array([9, 3, 1, 7, 4, 3, 6])
# # unsorted array print
# print('Original array:\n', a)
# # Sort array indices
# b = np.argsort(a)
# print('Sorted indices of original array->', b)
# # To get sorted array using sorted indices
# # c is temp array created of same len as of b
# c = np.zeros(len(b), dtype = int)
# print('Sorted array->', c)
# for i in range(0, len(b)):
#     c[i]= a[b[i]]
# print('Sorted array->', c)


# # Numpy array created
# # First column
# a = np.array([9, 3, 1, 3, 4, 3, 6])
# # Second column 
# b = np.array([4, 6, 9, 2, 1, 8, 7]) 
# print('column a, column b')
# for (i, j) in zip(a, b):
#     print(i, ' ', j)
# print("sorted index of a: ",np.argsort(a))
# print("sorted index of b: ",np.argsort(b))
# # Sort by a then by b
# ind = np.lexsort((b, a)) 
# print('Sorted indices->', ind)

'''
numpy.ndarray.sort()	Sort an array, in-place.
numpy.msort()	        Return a copy of an array sorted along the first axis.
numpy.sort_complex()	Sort a complex array using the real part first, then the imaginary part.
numpy.partition()	    Return a partitioned copy of an array.
numpy.argpartition()	Perform an indirect partition along the given axis using the algorithm specified by the kind keyword.
'''

# # Working on 2D array
# array = np.arange(12).reshape(3, 4)
# print("INPUT ARRAY : \n", array)
# # No axis mentioned, so works on entire array
# print("\nMax element : ", np.argmax(array))
# # returning Indices of the max element
# # as per the indices
# print(("\nIndices of Max element : ", np.argmax(array, axis=0)))
# print(("\nIndices of Max element : ", np.argmax(array, axis=1)))


# # Working on 1D array
# array = [np.nan, 1, 2, 3, 8]
# print("INPUT ARRAY 1 : \n", array)
# array2 = np.array([[np.nan, 1], [8, 3]])
# # returning Indices of the max element
# # as per the indices ingnoring NaN
# print("\nIndices of max in array1 : ", np.nanargmax(array))
# # Working on 2D array
# print("\nINPUT ARRAY 2 : \n", array2)
# print("\nIndices of max in array2 : ", np.nanargmax(array2))
# print("\nIndices at axis 1 of array2 : ", np.nanargmax(array2, axis = 1))


# # Working on 1D array
# array = np.arange(8)
# print("INPUT ARRAY : \n",array)
# # returning Indices of the min element
# # as per the indices
# print("\nIndices of min element : ", np.argmin(array, axis=0))

'''
numpy.nanargmin()	    Return the indices of the minimum values in the specified axis ignoring NaNs.
numpy.argwhere()	    Find the indices of array elements that are non-zero, grouped by element.
numpy.nonzero()	        Return the indices of the elements that are non-zero.
numpy.flatnonzero()	    Return indices that are non-zero in the flattened version of a.
numpy.where()	        Return elements chosen from x or y depending on condition.
numpy.searchsorted()	Find indices where elements should be inserted to maintain order.
numpy.extract()	        Return the elements of an array that satisfy some condition
'''
# Counting a number of 
# non-zero values
# a = np.count_nonzero([[0,1,7,0,20],
#                       [3,0,0,2,19]])
# b = np.count_nonzero([[0,1,7,0,30],
#                       [3,0,0,2,19]], axis=0)
# print("Number of nonzero values is :",a)
# print("Number of nonzero values is :",b)


# array = np.array([[1, 2], [3, 4]])
# # using flatten method
# array.flatten()
# print(array)
# #using fatten method 
# array.flatten('F')
# print(array)
















