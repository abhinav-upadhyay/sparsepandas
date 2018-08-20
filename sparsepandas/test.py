import pandas as pd
from base import SparseArrayType, SparseExtensionArray
import sparse
import numpy as np

def test_concat_columns(data ):
    df1 = pd.DataFrame({'A': data[:3]})
    df2 = pd.DataFrame({'B': [1, 2, 3]})

    expected = pd.DataFrame({'A': data[:3], 'B': [1, 2, 3]})
    result = pd.concat([df1, df2], axis=1)
    df2 = pd.DataFrame({'B': [1, 2, 3]}, index=[1, 2, 3])
    import pdb; pdb.set_trace()
    expected = pd.DataFrame({
        'A': data._from_sequence(list(data[:3]) + [np.nan]),
        'B': [np.nan, 1, 2, 3]})
    result = pd.concat([df1, df2], axis=1)
    print(expected.info())
    print(result.info())

arr = np.array(list(range(1, 101)))
sparray = SparseExtensionArray(arr)
test_concat_columns(sparray)

