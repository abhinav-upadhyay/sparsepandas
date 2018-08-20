An extension [array](https://pandas.pydata.org/pandas-docs/stable/extending.html) implementation for pandas to support sparse arrays using the [sparse](https://sparse.pydata.org/en/latest/index.html) module.

Inspired by the IPArray implementation from [cyberpandas](https://github.com/ContinuumIO/cyberpandas). Work in progress - at this moment 69 of the 86 Pandas interface tests are passing while 17 are failing.

Example Usage
```python

In [1]: import sparse

In [2]: import numpy as np

In [3]: import pandas as pd

In [4]: from sparse_array import SparseExtensionArray

In [5]: arr = np.random.random(20000)

In [6]: arr[arr<0.9] = 0.0

In [7]: sparse_arr = SparseExtensionArray(sparse_arr)

In [9]: df = pd.DataFrame({'sparse_col1': sparse_arr})

In [11]: df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 20000 entries, 0 to 19999
Data columns (total 1 columns):
sparse_col1    2062 non-null sparsetype
dtypes: sparsetype(1)
memory usage: 32.3 KB

In [12]: df.mean()
Out[12]:
sparse_col1    0.09797
dtype: float64

In [13]: %timeit df.sum()
1.6 ms ± 13.3 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

```
