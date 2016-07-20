---
layout: post
title:  "How to set up 32bit scikit-learn on Mac without additional installation "
date:   2016-07-17 18:52:21 -0500
categories: GSoC
comments: true
---

Sometimes you may want to know how scikit-learn behaves when it's running on 32-bit Python. This blog post try to give the simplest solution.

## Step by Step

Below I'll go through the procedure step by step:

1. Type the following command and make sure it outputs `2147483647`.

	```
	arch -32 /System/Library/Frameworks/Python.framework/Versions/2.7/bin/python -c "import sys; print sys.maxint"
	```

2. Modify **line 5** of `Makefile` exists in root directory of scikit-learn becomes: 

	```
	PYTHON ?= arch -32 /System/Library/Frameworks/Python.framework/Versions/2.7/bin/python
	```
	
	and modify **line 11** to:
	
	```
	BITS := $(shell PYTHON -c 'import struct; print(8 * 	struct.calcsize("P"))')
	```

3. Type

	```
	sudo make
	```
	
	in the root directory of scikit-learn and you are good to go!

## Verification

You can verify if 32-bit version of scikit-learn built successfully by typing:

```
arch -32 /System/Library/Frameworks/Python.framework/Versions/2.7/bin/python
```

to enter 32-bit Python shell.

After that, type:

```
import sklearn
```

to check if `sklearn` can now run on 32-bit Python.

Hope this helps!
