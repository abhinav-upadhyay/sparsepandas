from pandas.core.arrays import ExtensionArray, ExtensionOpsMixin
from collections import Iterable
from pandas.api.extensions import ExtensionDtype, take
from pandas.compat import set_function_name
from pandas.core.dtypes.generic import ABCSeries, ABCIndexClass
from pandas.core import ops
from pandas.core.dtypes.dtypes import registry
import sparse
import operator
import numpy as np
from numba import jit
from numba.types import float64, Tuple, int64, void


@jit(Tuple((int64[:, :], float64[:], int64))(int64[ :], float64[:], int64, int64, float64), parallel=True, nogil=True, nopython=True)
def _setitem(coords, data, shape, key, value):
    # TODO: This seems like a very slow way to do it
    data_iter = 0
    key_copied = False
    if value == 0.0:
        return np.expand_dims(coords, 0), data, shape
    newlen = data.shape[0] + 1
    new_coords = np.empty((1, newlen), dtype=np.int64)
    new_data = np.empty(newlen)
    for i in range(newlen):
        if data_iter >= 0:
            if not key_copied:
                if coords[data_iter] < key:
                    new_coords[0][i] = coords[data_iter]
                    new_data[i] = data[data_iter]
                    if data_iter < newlen - 2:
                        data_iter += 1
                    else:
                        data_iter = -1
                else:
                    new_coords[0][i] = key
                    new_data[i] = value
                    key_copied = True
            else:
                new_coords[0][i] = coords[data_iter]
                new_data[i] = data[data_iter]
                data_iter += 1
    if key >= data.shape[0]:
        new_shape = shape + 1
    else:
        new_shape = shape
    return new_coords, new_data, new_shape


class SparseArrayType(ExtensionDtype):
    name = 'sparsetype'
    type = float
    na_value = 0.0

    @classmethod
    def construct_from_string(cls, string):
        if string == cls.name:
            return cls()
        else:
            raise TypeError("Cannot construct a '{}' from "
                            "'{}'".format(cls, string))

    @property
    def _is_numeric(self):
        return True

    def construct_array_type(self, arg=None):
        if arg is not None:
            raise NotImplementedError()
        return SparseExtensionArray

class SparseExtensionArray(ExtensionArray, ExtensionOpsMixin):
    _dtype = SparseArrayType()
    ndim = 1
    can_hold_na = True
    kind = 'f'

    def __init__(self, coords, data=None, shape=None):
        if data is not None:
            self.data = sparse.COO(coords, data, fill_value=self.na_value, shape=shape)
        else:
            if isinstance(coords, sparse.COO):
                self.data = coords
            elif not isinstance(coords, np.ndarray):
                self.data = sparse.COO(np.array(coords, dtype=float), fill_value=self.na_value)
            else:
                self.data = sparse.COO(coords, fill_value=self.na_value)
    
    @classmethod
    def _from_ndarray(cls, data):
        return cls(data)

    @classmethod
    def _from_pyints(cls, values):
        return cls(np.array(values))
    
    @property
    def na_value(self):
        return self.dtype.na_value
    
    def __repr__(self):
        return "%s : %s" % (self.dtype.name, self.data.data)
    
    
    def _formatting_values(self):
        #return (np.array([x for x in self.data.data]))
        return self.data.todense()
    
    def _format(self, val):
        return "%s" % ((val))
    
    
    @staticmethod
    def _box_scalar(scalar):
        return scalar
    
    def _setitem(self, key, value):
        if isinstance(value, Iterable):
            new_coords, new_data, new_shape = _setitem(self.data.coords[0], self.data.data, self.shape[0], key, value[0])
        else:
            new_coords, new_data, new_shape = _setitem(self.data.coords[0], self.data.data, self.shape[0], key, value)
        self.data.coords = new_coords
        self.data.data = new_data
        self.data.shape = (new_shape,)
            
    def __setitem__(self, key, value):
        # TODO if value is self.na_value, remove that index from coords and data
        coords = self.data.coords.ravel()
        if isinstance(key, int):
            if key not in self.data.coords:
                self._setitem(key, value)
            else:
                if value == self.na_value:
                    return
                idx = np.where(coords == key)[0][0]
                self.data.data[idx] = value
        elif isinstance(key, slice):
            slice_start = key.start
            slice_end = key.stop
            slice_step = slice.step
            if slice_step and slice_step < 0:
                slice_step = np.abs(slice_step)
                slice_start, slice_send = slice_end, slice_start
            if slice_start and slice_start < 0:
                slice_start = self.shape[0] + slice_start
            if slice_end and slice_end < 0:
                slice_end = self.shape[0] + slice_end
            if slice_start and not slice_end:
                indices = key.indices(slice_start)
            else:
                indices = key.indices(slice_end)
            start, end, step = indices
            for i in range(start, end, step):
                if i not in coords:
                    self._setitem(i, value)
                else:
                    idx = np.where(coords == i)[0][0]
                    self.data.data[idx] = value
        elif isinstance(key, (np.ndarray)):
            if isinstance(key[0], np.bool_):
                for ind, v in enumerate(key):
                    if v:
                        if ind not in coords:
                            self._setitem(ind, value)
                        else:
                            idx = np.where(coords == i)[0][0]
                            self.data.data[idx] = value
            else:
                for k in key:
                    ind = k if k >= 0 else self.shape[0] + k
                    if ind not in coords:
                        self._setitem(ind, value)
                    else:
                        idx = np.where(coords == ind)[0][0]
                        self.data.data[idx] = value


    def __iter__(self):
        for x in iter(self.data.data, self.data.coords[0], self.shape[0], self.na_value):
            yield x

    @property
    def dtype(self):
        return self._dtype
    
    @classmethod
    def _from_sequence(cls, scalars, dtype=float, copy=False):
        if isinstance(dtype, SparseArrayType):
            dtype = float
        return SparseExtensionArray(np.array(scalars, dtype=dtype))
    
    @classmethod
    def _from_factorized(cls, values, original):
        return cls(values)
    
    def factorize(self, na_sentinel=-1):
        uniques = self.unique()
        unique_indices = {u:ind for ind, u in enumerate(uniques)}
        coords = self.data.coords.ravel()
        coords_set = set(coords.tolist())
        indices = [unique_indices[self.data.data[np.where(coords == i)[0][0]]] if i in coords_set else na_sentinel for i in range(self.shape[0]) ]
        return np.array(indices), uniques
    
    @property
    def shape(self):
        return (len(self.data),)
    
    def __len__(self):
        return self.shape[0]
    
    def __getitem__(self, item):
        coords = self.data.coords.ravel()
        if isinstance(item, int):
            if item < 0:
                item = self.shape[0] + item
            if item < self.shape[0] and item not in self.data.coords:
                return self.na_value
            data_row = np.where(coords == item)[0][0]
            return (self.data.data[data_row])
        elif isinstance(item, slice):
            slice_start = item.start
            slice_stop = item.stop
            if slice_start and slice_start < 0:
                #slice_start = self.data.shape[0] + slice_start
                slice_start = self.shape[0]
            if slice_stop and slice_stop < 0:
                slice_stop = self.data.shape[0] + slice_stop
            if slice_start and not slice_stop:
                indices = item.indices(slice_start)
            elif slice_stop:
                indices = item.indices(item.stop)
            else:
                indices = item.indices(self.shape[0])
            start, end, step = indices
            
            values = []
            for i in range(start, end, step):
                if i not in coords:
                    values.append(self.na_value)
                else:
                    idx = np.where(coords == i)[0]
                    values.append(self.data.data[idx][0])
            return type(self)(self._from_sequence(values))
        elif isinstance(item, (np.ndarray, list)):
            values = []
            if len(item) > 0 and isinstance(item[0], np.bool_):
                for ind, value in enumerate(item):
                    if value:
                        if ind < 0:
                            ind = self.data.shape[0] + ind
                        if ind not in coords:
                            values.append(self.na_value)
                        else:
                            idx = np.where(coords == ind)[0]
                            values.append(self.data.data[idx][0])
                return type(self)(self._from_sequence(values))
            else:
                for ind in item:
                    ind = ind if ind >= 0 else self.shape[0] + ind
                    if ind not in coords:
                        values.append(self.na_value)
                    else:
                        idx = np.where(coords == ind)[0]
                        values.append(self.data.data[idx][0])
                return type(self)(self._from_sequence(values))

    
    def take(self, indices, allow_fill=False, fill_value=None):
        if self.shape[0] == 0 and not allow_fill:
            raise IndexError("cannot do a non-empty take")
            
        indices = np.asarray(indices, dtype='int')
        if allow_fill and fill_value is None:
            fill_value = self.na_value

        if allow_fill:
            if not len(self):
                if not (indices == -1).all():
                    msg = "Invalid take for empty array. Must be all -1."
                    raise IndexError(msg)
                else:
                    # all NA take from and empty array
                    took = (np.full((len(indices), 2), fill_value, dtype=float)
                              .reshape(-1))
                    return self._from_ndarray(took)
            if (indices < -1).any():
                msg = ("Invalid value in 'indicies'. Must be all >= -1 "
                       "for 'allow_fill=True'")
                raise ValueError(msg)

        indexer = []
        for x in indices:
            if x >= self.shape[0]:
                raise IndexError("out of bounds")
            indexer.append(x)
        took = np.full((len(indexer)), fill_value, dtype=float)


        for i, ind in enumerate(indexer):
            coords = self.data.coords.ravel()
            if ind == -1 and allow_fill:
                took[i] = fill_value
            elif ind < 0 and not allow_fill:
                ind = self.data.shape[0] + ind
                if ind in coords:
                    data_row = np.where(coords == ind)[0][0]
                    took[i] = self.data.data[data_row]
                else:
                    took[i] = fill_value
            elif ind not in coords:
                took[i] = self.na_value
            else:
                data_row = np.where(coords == ind)[0][0]
                took[i] = self.data.data[data_row]
        return type(self)(took)
    
    def take_nd(self, indexer, allow_fill=True, fill_value=None):
        return self.take(indexer, allow_fill=allow_fill, fill_value=fill_value)

    def isna(self):
        nas = np.ones(self.data.shape, dtype=bool)
        nas[self.data.coords[0, :]] = False
        return nas

    @property
    def nbytes(self):
        return self.data.nbytes

    def copy(self, deep=False):
        return SparseExtensionArray(self.data.coords.copy(), self.data.data.copy(), shape=self.data.shape)

    @classmethod
    def _concat_same_type(cls, to_concat):
        return SparseExtensionArray(coords=sparse.concatenate([array.data for array in to_concat]))

    def tolist(self):
        return self.data.todense().tolist()

    def argsort(self, axis=-1, kind='quicksort', order=None):
        return self.data.coords[0, self.data.data.argsort()]

    def unique(self):
        # type: () -> ExtensionArray
        # https://github.com/pandas-dev/pandas/pull/19869
        return self._from_sequence(np.unique(self.data.data))
    
    def __lt__(self, other):
        return self.data < other

    def __le__(self, other):
        return self.data <= other

    def __eq__(self, other):
        return self.data.data == other.data.data and self.data.coords == other.data.coords

    def __ge__(self, other):
        return other <= self

    def __gt__(self, other):
        return other < self

    def equals(self, other):
        if not isinstance(other, type(self)):
            raise TypeError
        return (self.data == other.data).all()
    
    def astype(self, dtype, copy=True):
        return np.array(self.data.todense(), copy=copy, dtype=dtype)

    @classmethod
    def _create_arithmetic_method(cls, op):
        #print('op name: %s' % op.__name__)
        def sparse_arithmetic_method(self, other):
            op_name = op.__name__
            if op_name == 'add':
                coords,data = _sum2(self.data.coords[0], self.data.data, other.values.data.coords[0], other.values.data.data)
                coo = sparse.COO(coords, data, shape=self.shape)
                return SparseExtensionArray(coo)

        opname = ops._get_op_name(op, True)
        return set_function_name(sparse_arithmetic_method, opname, cls)

@jit(Tuple((int64[:, :], float64[:]))(int64[ :], float64[:], int64[ :], float64[:]),  nogil=True, nopython=True)
def _sum2(coords1, data1, coords2, data2):
    iter1 = 0
    iter2 = 0
    len1 = coords1.shape[0]
    len2 = coords2.shape[0]
    out_coords = np.empty(len1 + len2, dtype=np.int64)
    out_data = np.empty(len1 + len2)
    index = 0
    while not (iter1 == len1 and iter2 == len2):
        if iter1 < len1 and iter2 < len2:
            c1 = coords1[iter1]
            c2 = coords2[iter2]
            if c1 == c2:
                out_coords[index] = c1
                out_data[index] = data2[iter2] + data1[iter1]
                index += 1
                iter1 += 1
                iter2 +=1
            elif c1 < c2:
                out_coords[index] = c1
                out_data[index] = data1[iter1]
                index += 1
                iter1 += 1
            else:
                out_coords[index] = c2
                out_data[index] = data2[iter2]
                index += 1
                iter2 += 1
        elif iter1 < len1:
            end_index = index + len1 - iter1
            out_coords[index: end_index] = coords1[iter1:]
            out_data[index: end_index] = data1[iter1:]
            index = end_index
            break
        else:
            end_index = index + len2 - iter2
            out_coords[index: end_index] = coords2[iter2:]
            out_data[index: end_index] = data2[iter2:]
            index = end_index
            break
    out_coords = np.expand_dims(out_coords[:index], 0)
    out_data = out_data[:index]
    return out_coords, out_data



@jit(Tuple((int64[:, :], float64[:]))(int64[ :], float64[:], int64[ :], float64[:]),  nogil=True, nopython=True)
def _sum(coords1, data1, coords2, data2):
    coords1_set = set(coords1)
    out_coords = []
    common_coords = []
    out_data = []
    for i, c in enumerate(coords2):
        if c in coords1_set:
            index = np.where(coords1 == c)[0][0]
            out_data.append(data1[index] + data2[i])
            common_coords.append(c)
        else:
            out_data.append(data2[i])
        out_coords.append(c)

    common_coords_set = set(common_coords)
    for i, c in enumerate(coords1):
        if c in common_coords_set:
            continue
        out_data.append(data1[i])
        out_coords.append(c)

    out_coords = np.array(out_coords)
    out_coords = np.expand_dims(out_coords, 0)
    return out_coords, np.array(out_data)


SparseExtensionArray._add_arithmetic_ops()
registry.register(SparseArrayType)
#SparseExtensionArray._add_comparison_ops()


@jit(void(float64[:], int64[:], int64, float64), parallel=True, nogil=True, nopython=True)
def iter(data, coords, length, na_value):
    coords_len = data.shape[0]
    coords_iter = 0
    for i in range(length):
        if coords_iter >= coords_len:
            yield na_value
        elif coords[coords_iter] != i:
            yield na_value
        else:
            val = data[coords_iter]
            coords_iter += 1
            yield val
    