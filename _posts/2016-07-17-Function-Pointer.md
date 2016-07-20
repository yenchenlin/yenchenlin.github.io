---
layout: post
title:  "Using Function Pointer to Maximize Code Reusability in Cython"
date:   2016-07-20 18:52:21 -0500
categories: GSoC
comments: true
---

When writing C, function pointer is extremely useful because it can help us define a **callback** function, i.e., **a way to parametrize a function**. This means that some part of the function behavior is not hard-coded into itself, but into the callback function provided by user. Callers can make function behave differently by passing different callback functions. A classic example is `qsort()` from the C standard library that takes its sorting criterion as a pointer to a comparison function.

Besides the benefit above, we can also use function pointer straightforwardly to avoid redundant control flow code such as `if`, `else`.

In this blog post, I'm going to explain how we can combine function pointer and Cython fused types in a easy way to make function pointer become more powerful than ever, and therefore maximize the code reusability in Cython.

## Function Pointer

Let's start from why function pointer can help us address code duplication issue.

Consider we have the following two functions, one add one and the other add two to the function argument:

```c
float add_one(float x) {
	return x+1;
}

float add_two(float x) {
	return x+2;
}
```
Now close your eyes, try your best to imagine the operation `x+1` performed in `add_one` and the operation `x+2` performed in `add_two` are costly which must be implemented in C or they will take several hours to complete. 

Okay, base on the <del>imagined</del> reason above, we indeed need to import C funcitons above to speed up our Cython function, which will return `(x+1)*2+1` if `x` is an odd number, or `(x+2)*2+2` if `x` is an even number:

```python
cdef float linear_transform(float x):
	"""
	This function will return (x+1)*2+1 if x is odd
	                          (x+2)*2+2 if x is even
	"""
	
	float ans

	if x % 2 == 1: # x is odd
		ans = add_one(x)
	else:          # x is even
		ans = add_two(x)
	
	ans *= 2
	
	// Code duplication
	if x % 2 == 1:
		ans = add_one(x)
	else:
		ans = add_two(x)
	
	return ans
```

As one can see, there is a code duplication appears in the end of this function, because we have to check whether we need to apply `add_one` or `add_two` to the variable `x`.

To address this issue, we can definine a function pointer and assign it once we know `x` is a odd number or even number. Until the end of the program, we don't have to write annoying `if`, `else` again.

Above code snippet can reduce to:

```python
ctypedef float (*ADD)(float x)

cdef float linear_transform(float x):
	"""
	This function will return (x+1)*2+1 if x is odd
	                          (x+2)*2+2 if x is even
	"""
	ADD add
	float ans

	if x % 2 == 1: # x is odd
		add = add_one
	else:          # x is even
		add = add_two
	
	ans *= 2
	ans = add(ans)
	
	return ans
```

This code snippet is more readable. Function pointer do make our code looks neat!

**Note**: Although there is only one duplication in above example, there may be a lot in real code, which can show function pointer's value more obviously.

## Function Pointer's Limitation

However, function pointer is not omnipotent. Although they provide a good way to write generic code, unfortunately they don't provide you with **type generality**. What do I mean?

Consider if we now have the following two C functions that both add 1 to the argument variable, one is for type `float` and one is for type `double`:

```c
float add_one_float(float x) {
	return x+1;
}

double add_one_double(double x) {
	return x+1;
}
```

Now do the imagination process described in our first example again, base on the same reason, we indeed need to load these extern C functions to speed up the following Cython function `linear_transform `:

```python
cdef floating linear_transform(floating x):
	"""
	This function will return (x+1)*2+1 with the 
	same type as input argument
	"""
	
	floating ans

	if floating is float:
		ans  = add_one_float(x)
	elif floating is double:
		ans  = add_one_double(x)
	
	ans *= 2
	
	if floating is float:
		ans  = add_one_float(x)
	elif floating is double:
		ans  = add_one_double(x)
	
	return ans
```

Don't be scared if you havn't seem `floating` before, to be brief, `floating` here refers to either type `float` **or** type `double`.
It is just a feature called [fused types](http://docs.cython.org/src/userguide/fusedtypes.html) in Cython, which basically serves the same role like templates in C++ or generics in Java. 

Note that now we can't define a function pointer and assign it like what we did in our first example, because C functions `add_one_float` and `add_one_double` have different function signatures. Since C is a strong typed language, it's hard to define a function pointer that can point to functions with different types. (which is why, for example, the standard library `qsort` still requires a function that takes `void*` pointer.)

**NOTE**: Usage of `void*` pointer in C is beyond the scope of this blog post, you can find a simple introduction [here](http://www.circuitstoday.com/void-pointers-in-c). But remember, it's dangerous.

## Finction Pointer + Fused Types

Fortunately, [fused types](http://docs.cython.org/src/userguide/fusedtypes.html) is here to rescue us. With this useful tool, we can actually define fused types function pointer to solve above problem!

```python
ctypedef floating (*ADD)(floating x)

cdef floating linear_transform(floating x):
	"""
	This function will return (x+1)*2+1 with the 
	same type as input argument
	"""
	
	ADD add_one
	floating ans

	if floating is float:
		add_one = add_one_float
	elif floating is double:
		add_one = add_one_double
		
	ans = add_one(x) # (x+1)

	ans *= 2         # (x+1)*2
	
	ans = add_one(x) # (x+1)*2+1
	
	return ans
```

Note that since `floating` can represent either `float` or `double`, function pointer of type `floating` have the ability to achieve **type generality**, which is not available before we combine fused types with function pointer. 

Finally, we are going to demystify the secret of this magic trick performed by Cython and make sure that it works properly.

## Demystifying How It Work

In order to know how Cython fused types function pointer works, let's become a ninja and dive deep to peep the C code generated by Cython.

In the generated C code of above Cython function, there is no `if floating is float:` anymore. Actually, to accommodate fused types `floating`, Cython generates one version of the function for `float` and another for `double`.

And in the `float` version function, it directly assign the function pointer we declared to the actual C function that will be called if `x` is of type `float`:

```c
	__pyx_v_add_one = add_one_float;
```

Same as the `float` version, `double` version also generates

```c
	__pyx_v_add_one = add_one_double;
```

which directly assigns function pointer to the correct function.

In fact, this direct assignment is an optimization performed by C compiler since it can identify variables that remain unchanged within a function. It would find out that function pointer `__pyx_v_add_one` is only set once to a constant, i.e., an extern function. Hence after object code is linked, `__pyx_v_add_one ` will directly be assigned to the C function.

On the contrary, Python interpreter can provides only little in static analysis and code optimization since the language design doesn't have the *compile* phase.

In sum, always implement your computation heavy code in Cython instead of Python.

## Summary

Combining function pointer with fused types raises its power to another higher level. Actually, it is a generalized version of original function pointers, and can be used in lots of places to make our code looks more readable and cleaner.
Also, it is often a good idea to check the C code generated by Cython so as to make sure it's doing what you hoped.

See you next time!
