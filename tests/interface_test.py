import pytest

from pandas.tests.extension import base
import pandas.util.testing as tm

from sparsepandas.sparse_array import SparseExtensionArray, SparseArrayType
import numpy as np
from sparsepandas.sparse_array import (
    Int8Dtype, Int16Dtype, Int32Dtype, Int64Dtype,
    UInt8Dtype, UInt16Dtype, UInt32Dtype, UInt64Dtype, Float64Dtype, Float32Dtype)

class BaseSparseArray(object):
    def assert_series_equal(self, left, right, *args, **kwargs):
        if isinstance(left.dtype, SparseArrayType) and isinstance(right.dtype, SparseArrayType):
            return left.data.shape == right.data.shape and left.data.nnz == right.data.nnz and np.all(left.data.data == right.data.data) and np.all(left.data.coords == right.data.coords)
        return left.shape == right.shape and left.values == right.values


@pytest.fixture(params=[Float64Dtype, Int8Dtype, Int16Dtype, Int32Dtype, Int64Dtype, Float32Dtype,
                        UInt8Dtype, UInt16Dtype, UInt32Dtype, UInt64Dtype])
def dtype(request):
    return request.param()


@pytest.fixture
def data(dtype):
    d = np.arange(1, 101, dtype=np.float64)
    return SparseExtensionArray(d, dtype=dtype)


@pytest.fixture
def data_missing():
    arr = np.array([na_value(), 3.0])
    return SparseExtensionArray(arr)

@pytest.fixture
def data_repeated(data):
    def gen(count):
        for _ in range(count):
            yield data
    yield gen


@pytest.fixture(params=['data', 'data_missing'])
def all_data(request, data, data_missing):
    """Parametrized fixture giving 'data' and 'data_missing'"""
    if request.param == 'data':
        return data
    elif request.param == 'data_missing':
        return data_missing


@pytest.fixture
def data_for_sorting():
    return SparseExtensionArray([10, 2 ** 64 - 1, 1.0])


@pytest.fixture
def data_missing_for_sorting():
    return SparseExtensionArray([2 ** 64 - 1, na_value(), 1.0])


@pytest.fixture
def data_for_grouping():
    b = 1.0
    a = 2 ** 32 + 1
    c = 2 ** 32 + 10
    return SparseExtensionArray([
        b, b, na_value(), na_value(), a, a, b, c
    ])


@pytest.fixture
def na_cmp():
    """Binary operator for comparing NA values.

    Should return a function of two arguments that returns
    True if both arguments are (scalar) NA for your type.

    By defult, uses ``operator.or``
    """
    return lambda x, y: int(x) == int(y) == 0
    #return lambda x, y: np.isnan(x) and np.isnan(y)


@pytest.fixture
def na_value():
    return SparseArrayType.na_value


class TestDtype(BaseSparseArray, base.BaseDtypeTests):
    @pytest.mark.skip(reason="using multiple dtypes")
    def test_is_dtype_unboxes_dtype(self):
        pass


class TestInterface(base.BaseInterfaceTests):
    pass


class TestConstructors(base.BaseConstructorsTests):
    pass


class TestReshaping(base.BaseReshapingTests):
    @pytest.mark.skip(reason='Pandas inferrs us as int64.')
    def test_concat_mixed_dtypes(self):
        pass
    
    #@pytest.mark.skip(reason='Pandas inferrs us as int64.')
    #def test_concat_columns(self):
    #    pass



class TestGetitem(base.BaseGetitemTests):
    pass


class TestMissing(base.BaseMissingTests):
    @pytest.mark.skip(reason='pandas expects to call nonzero on extension array, we need to revisit this')
    def test_dropna_frame(self):
        pass


class TestMethods(BaseSparseArray, base.BaseMethodsTests):
    @pytest.mark.xfail(reason='upstream')
    def test_value_counts(self, data, dropna):
        pass

