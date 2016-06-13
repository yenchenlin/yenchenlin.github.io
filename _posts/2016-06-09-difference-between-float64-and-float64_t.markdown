---
layout: post
title:  "Difference between np.float64 & np.float64_t"
date:   2016-06-08 18:52:21 -0500
categories: GSoC
comments: true
---

If you are a newcomer to Cython just like me, it is probably that you will be confused by the usage time of `np.float64_t` and `np.float64`.

Below, I'll briefly introduce how these two types are fundamentally different, and generalize this concept to other datatypes as well.

Before that, we need to know a bit of Cython.

## cimport in Cython

In Python, we use the `import` statement to access functions, objects, and classes inside other _modules_ and _packages_.

Cython also fully supports the `import` statement, so it allows us to access Python objects defined in external Python modules.

However, note that if above were the end of the story, Cython modules would not allow to access other Cython modules' `cdef` functions, `ctypedef`s, or structs, and it would not allow C-level access to other extension types.

To remedy this, Cython has a `cimport` statement that provides **compile time** access to C-level constructs, and it looks for these constructs’ declarations from separate Cython files called _definition files_, which have a _.pxd_ extension and need to be created by us.

In a _.pxd_ file, we can place the _declarations_ of C-level constructs that we wish to share, and only the _declarations_ here are cimportable. Also, since `some_file_name.pxd` created by us have the same base name as the original file `some_file_name.pyx`, they are treated as one namespace by Cython. Therefore, we need to modify `some_file_name.pyx` in order to remove the repeat declarations in it.

And after we have created the _.pxd_ file and clear the _.pyx_ file, now an external implementation file can access all C-level constructs inside _.pyx_ via the `cimport` statement.

Let's take an real-world example to see how `cimport` works!

## cimport numpy as np

In some files of the well-known machine library scikit-learn (such as [this one](https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/utils/sparsefuncs_fast.pyx)), you can find the following code snippet:

```python
cimport numpy as np
import numpy as np
``` 

I remember I was stunned when I saw these lines for the first time, WHAT IS IT?

Well, the good news is that we now know the basics of the `cimport` statement, so we can figure it out step by step.

First, since only the declarations in _.pxd_ file are cimportable, we have to identify which _.pxd_ file is Cython looking when executing `cimport numpy as np`.

After a bit of research, we should find that a file called [`__init__.pxd`](https://github.com/cython/cython/blob/970c2fc0e676ca22016e14147ada0edba937dc6b/Cython/Includes/numpy/__init__.pxd) lies in the [**numpy** folder](https://github.com/cython/cython/tree/master/Cython/Includes/numpy) under our Cython installation. An `__init__.pxd` file can make Cython treat the directory as a package just like how `__init.py__` works for Python (see [here](http://stackoverflow.com/questions/448271/what-is-init-py-for)). Therefore, in this case Cython will treat the **numpy** folder as a package and give us access to Numpy's C API defined in the [`__init__.pxd`](https://github.com/cython/cython/blob/970c2fc0e676ca22016e14147ada0edba937dc6b/Cython/Includes/numpy/__init__.pxd) file during **compile time**.

On the contrary, `import numpy as np` will only give us access to Numpy's pure-Python API and it occurs at **runtime**.

Note that here we use the same alias (i.e., `np`) for both of the imported external packages, but thanks to the almighty Cython which will internally handles this ambiguity, we don't not need to use different names.

## np.float64 v.s np.float64_t

So here comes our main topic, what is the difference between `np.float64` and `np.float64_t`, and which should I use?

#### np.float64

`np.float64` is a Python `type` object that is defined at Python level to represent 64 bits float data, and it has common attributes such as `__name__` that most of other Python objects have too. You can simply use the following code to verify it:

```python
import numpy as np
type(np.float64)
print np.float64
print np.float64.__name__
``` 

#### np.float64

In [`__init__.pxd`](https://github.com/cython/cython/blob/970c2fc0e676ca22016e14147ada0edba937dc6b/Cython/Includes/numpy/__init__.pxd), you can find the following lines:

```python
ctypedef double       npy_float64
ctypedef npy_float64    float64_t
```

So it is clear that `np.float64_t` represents the type `double` in C, and it is nowhere near as a Python object. Therefore, if you call `print np.float64_t` in a _.pyx_ file, it will warn you the following message during compile time:
`'float64_t' is not a constant, variable or function identifier` 

#### Which to use?

Let's take another simple example to illustrate the usage time between these two types:

```python
import numpy as np
cimport numpy as np

def test():
	
	// 1
	cdef np.ndarray[np.float64_t, ndim=1] array
	
	// 2
	array = np.empty(10, dtype=np.float64)
	
	print array
```

1. We use `np.ndarray` to declare the type of the object exposing the buffer interface, and place C data type inside the bracket for the array elements. So, we should make sure we use `np.float64_t` here to specify the element's data type . 

2. To initialize the Numpy buffer we just declared, we can create an array object at Python level and assign it to the Numpy buffer. In this case, we should use `np.float64` since we are not declaring C type variable.

Of course, The same concept can be generalized to other data types (e.g., `np.int32` v.s `np.int32_t`, `np.int64` v.s `np.int_64_t`, etc.)

## Summary

After working on Cython for a month, I found debugging in Cython is both hard and frustrated because the documents is not really thorough.

Consequently, I hope this blog post can safe your effort by helping you clarify the difference between data types defined in Cython and Python. 

In the future, I will also document more of my findings about Cython during my GSoC.