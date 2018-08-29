import pytest

from pandas.tests.extension import base

from sparsepandas.sparse_array import SparseExtensionArray, SparseArrayType
import numpy as np


@pytest.fixture
def dtype():
    return SparseArrayType()


@pytest.fixture
def data():
    return SparseExtensionArray(list(range(1, 101)))


@pytest.fixture
def data_missing():
    arr = np.array([0.0, 3.0])
    #arr[0] = np.nan
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
    return SparseExtensionArray([2 ** 64 - 1, 0.0, 1])


@pytest.fixture
def data_for_grouping():
    b = 1
    a = 2 ** 32 + 1
    c = 2 ** 32 + 10
    return SparseExtensionArray([
        b, b, 0.0, 0.0, a, a, b, c
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


class TestDtype(base.BaseDtypeTests):
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
    pass


class TestMethods(base.BaseMethodsTests):
    @pytest.mark.xfail(reason='upstream')
    def test_value_counts(self, data, dropna):
        pass
