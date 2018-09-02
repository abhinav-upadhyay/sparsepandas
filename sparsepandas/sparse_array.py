import sys
from pandas.core.arrays import ExtensionArray, ExtensionOpsMixin
from collections import Iterable
from pandas.api.extensions import ExtensionDtype, take
from pandas.compat import set_function_name
from pandas.core.dtypes.generic import ABCSeries, ABCIndexClass
from pandas.core import ops
from pandas.core.dtypes.dtypes import registry
from pandas.core.dtypes.common import is_list_like, is_integer_dtype
import sparse
import operator
import numpy as np
from numba import jit
from numba.types import float64, Tuple, int64, void


@jit(Tuple((int64[:, :], float64[:], int64))(int64[ :], float64[:], int64, int64, float64), parallel=True, nogil=True, nopython=True)
#@jit(parallel=True, nogil=True, nopython=True)
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
    na_value = np.float(0.0)

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

    @property
    def numpy_dtype(self):
        """ Return an instance of our numpy dtype """
        return np.dtype(self.type)

    @property
    def kind(self):
        return self.numpy_dtype.kind


class SparseExtensionArray(ExtensionArray, ExtensionOpsMixin):
    ndim = 1
    can_hold_na = True

    def __init__(self, coords, data=None, dtype=None, shape=None):
        if dtype is None and hasattr(data, 'dtype'):
            if is_integer_dtype(data.dtype):
                dtype = data.dtype

        dtype = self._handle_dtype(dtype)
        if data is not None:
            self.data = sparse.COO(coords, data, fill_value=self.na_value, shape=shape)
        else:
            if isinstance(coords, sparse.COO):
                self.data = coords
            elif not isinstance(coords, np.ndarray):
                self.data = sparse.COO(np.array(coords, dtype=dtype), fill_value=self.na_value)
            else:
                self.data = sparse.COO(coords, fill_value=self.na_value)
    
    @classmethod
    def _from_ndarray(cls, data):
        return cls(data)

    @classmethod
    def _handle_dtype(cls, dtype):
        if dtype is not None:
            if not issubclass(type(dtype), SparseArrayType):
                try:
                    #dtype = _dtypes[str(np.dtype(dtype))]
                    dtype= SparseArrayType
                except KeyError:
                    raise ValueError("invalid dtype specified {}".format(dtype))
        if dtype is not None:
            dtype = dtype.numpy_dtype
        return dtype


    @classmethod
    def _from_pyints(cls, values):
        return cls(np.array(values))
    
    @property
    def na_value(self):
        return SparseArrayType.na_value
    
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
        return _dtypes[str(self.data.dtype)]
    
    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        dtype = cls._handle_dtype(dtype)
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

        coords = self.data.coords.ravel()
        for i, ind in enumerate(indexer):
            #TODO we don't need to set took[i] = fill_value as the array is initialised with fill_value
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
        return self.data < other.data

    def __le__(self, other):
        return self.data <= other.data

    def __eq__(self, other):
        return self.data.data == other.data.data and self.data.coords == other.data.coords

    def __ge__(self, other):
        return other.data <= self.data

    def __gt__(self, other):
        return other.data < self.data

    def equals(self, other):
        if not isinstance(other, type(self)):
            raise TypeError
        return (self.data == other.data).all()
    
    def astype(self, dtype, copy=True):
        return np.array(self.data.todense(), copy=copy, dtype=dtype)

    @classmethod
    def _create_comparison_method(cls, op):
        op_name = op.__name__
        #print(op_name)
        def cmp_method(self, other):
            if is_list_like(other):
                other = cls._from_sequence(other)

            if other.ndim > 0 and len(self) != len(other):
                raise ValueError("Lengths must match to compare")

            if op_name == 'eq':
                return SparseExtensionArray(self.data == other.data)
            if op_name == 'ne':
                return SparseExtensionArray(self.data != other.data)
            if op_name == 'lt':
                return SparseExtensionArray(self.data < other.data)
            if op_name == 'gt':
                return SparseExtensionArray(self.data > other.data)
            if op_name == 'le':
                return SparseExtensionArray(self.data <= other.data)
            if op_name == 'ge':
                return SparseExtensionArray(self.data >= other.data)

        opname = ops._get_op_name(op, True)
        return set_function_name(cmp_method, opname, cls)



    @classmethod
    def _create_arithmetic_method(cls, op):
        #print('op name: %s' % op.__name__)
        def sparse_arithmetic_method(self, other):
            op_name = op.__name__
            if op_name == 'add':
                coords,data = _sum(self.data.coords[0], self.data.data, other.values.data.coords[0], other.values.data.data)
                coo = sparse.COO(coords, data, shape=self.shape)
                return SparseExtensionArray(coo)

        opname = ops._get_op_name(op, True)
        return set_function_name(sparse_arithmetic_method, opname, cls)


@jit(nogil=True, nopython=True, cache=True, error_model='numpy')
def _sum(coords1, data1, coords2, data2):
    iter1 = 0
    iter2 = 0
    len1 = coords1.shape[0]
    len2 = coords2.shape[0]
    alloc_length = len1 + len2
    ret_out_coords = np.empty((1, alloc_length), dtype=np.int64)
    out_coords = ret_out_coords[0]
    out_data = np.empty(alloc_length)
    c1 = coords1[0]
    c2 = coords2[0]
    for index in range(alloc_length):
        if iter1 < len1 and iter2 < len2:
            #c1 = coords1[iter1]
            #c2 = coords2[iter2]
            if c1 < c2:
                out_coords[index] = c1
                out_data[index] = data1[iter1]
                iter1 += 1
                c1 = coords1[iter1] if iter1 < len1 else c1
            elif c2 < c1 > 0:
                out_coords[index] = c2
                out_data[index] = data2[iter2]
                iter2 += 1
                c2 = coords2[iter2] if iter2 < len2 else c2
            else:
                out_coords[index] = c1
                out_data[index] = data2[iter2] + data1[iter1]
                iter1 += 1
                iter2 += 1
                c1 = coords1[iter1] if iter1 < len1 else c1
                c2 = coords2[iter2] if iter2 < len2 else c2
        elif iter1 <= len1:
            end_index = index + len1 - iter1
            out_coords[index: end_index] = coords1[iter1:]
            out_data[index: end_index] = data1[iter1:]
            return ret_out_coords[:, :end_index], out_data[:end_index]
        else:
            end_index = index + len2 - iter2
            out_coords[index: end_index] = coords2[iter2:]
            out_data[index: end_index] = data2[iter2:]
            return ret_out_coords[:, :end_index], out_data[:end_index]
    #out_data = out_data[:index]
    #return ret_out_coords[:, :index], out_data

#@profile
#@jit(nogil=True, nopython=True, cache=True, error_model='numpy')
def _sum6(coords1, data1, coords2, data2):
    len1 = coords1.shape[0]
    len2 = coords2.shape[0]
    alloc_length = len1 + len2
    ret_out_coords = np.empty((1, alloc_length), dtype=np.int64)
    out_data = np.empty(alloc_length)
    out_coords = ret_out_coords[0]
    temp_out_coords = np.empty(alloc_length)
    temp_out_data = np.empty(alloc_length)
    temp_out_coords[:len1] = coords1[0:]
    temp_out_data[:len1] = data1[:]
    temp_out_coords[len1:] = coords2[0:]
    temp_out_data[len1:] = data2[0:]
    sorted_idx = np.argsort(temp_out_coords)

    index = 0
    i = 0
    while i < len(sorted_idx) - 1:
        idx = sorted_idx[i]
        if idx !=  temp_out_coords[idx] == temp_out_coords[idx + 1]:
            out_coords[index] = temp_out_coords[idx]
            out_data[index] = temp_out_data[idx] + temp_out_data[idx + 1]
            index += 1
            i += 2
        else:
            out_coords[index] = temp_out_coords[idx]
            out_data[index] = temp_out_data[idx]
            i += 1
            index += 1
    return ret_out_coords[:, :index], out_data[:index]


#@profile
@jit(nogil=True, nopython=True, cache=True, error_model='numpy')
def _sum5(coords1, data1, coords2, data2):
    len1 = coords1.shape[0]
    len2 = coords2.shape[0]
    alloc_length = len1 + len2
    ret_out_coords = np.empty((1, alloc_length), dtype=np.int64)
    out_data = np.empty(alloc_length)
    out_coords = ret_out_coords[0]
    out_coords[:len1] = coords1[0:]
    out_data[:len1] = data1[:]
    out_coords[len1:] = coords2[0:]
    out_data[len1:] = data2[0:]
    #sorted_idx = np.argsort(out_coords)
    #out_data = out_data[sorted_idx]
    #out_coords = out_coords[sorted_idx]
    index = 0
    iter1 = 0
    iter2 = len1
    c1 = out_coords[iter1]
    d1 = out_data[iter1]
    c2 = out_coords[iter2]
    d2 = out_data[iter2]
    temp_coords = np.zeros(len1, dtype=np.int64)
    temp_data = np.zeros(len1, dtype=np.float64)
    temp_ind = 0
    last_coord = max(coords1[-1], coords2[-1])
    while True:
        if c1 < c2:
            if temp_coords[temp_ind] != 0:
                out_coords[index] = temp_coords[temp_ind]
                out_data[index] = temp_data[temp_ind]
                temp_ind += 1
            index += 1
            if temp_coords[temp_ind] != 0:
                c1 = temp_coords[temp_ind]
                d1 = temp_data[temp_ind]
            else:
                iter1 += 1
                c1 = out_coords[iter1]
                d1 = out_data[iter1]
            if c1 == last_coord or c2 == last_coord:
                break
            continue
        if c1 > c2:
            if out_coords[index] != c2:
                temp_coords[temp_ind] = out_coords[index]
                temp_data[temp_ind] = out_data[index]    
                temp_ind += 1
            out_coords[index] = c2
            out_data[index] = d2
            iter1 += 1
            out_coords[index] = c2
            out_data[index] = d2
            iter2 += 1
            index += 1
            c2 = out_coords[iter2]
            d2 = out_data[iter2]
            if c2 == last_coord:
                break
            continue
        if c1 == c2:
            out_data[index] = d1 + d2
            iter1 += 1
            iter2 += 1
            index += 1
            c2 = out_coords[iter2]
            d2 = out_data[iter2]
            if temp_coords[temp_ind] == 0:
                c1 = out_coords[iter1]
                d1 = out_data[iter1]
            else:
                c1 = temp_coords[temp_ind]
                d1 = temp_data[temp_ind]
                temp_ind += 1
            if c1 == last_coord or c2 == last_coord:
                break
    
    while temp_ind != len1 or temp_coords[temp_ind] != 0:
        out_coords[index] = temp_coords[temp_ind]
        out_data[index] = temp_data[temp_ind]
        index += 1
        temp_ind += 1

        
    return ret_out_coords[:, :index], out_data[:index]

@jit(nogil=True, nopython=True, cache=True, error_model='numpy')
def _sum4(coords1, data1, coords2, data2):
    #import pdb; pdb.set_trace()
    iter1 = 0
    iter2 = 0
    len1 = coords1.shape[0]
    len2 = coords2.shape[0]
    alloc_length = len1 + len2
    ret_out_coords = np.empty((1, alloc_length), dtype=np.int64)
    out_coords = ret_out_coords[0]
    out_data = np.empty(alloc_length)
    index = 0
    c1 = coords1[0]
    c2 = coords2[0]
    # print('len1')
    # print(len1)
    # print('len2')
    # print(len2)
    while iter1 <= len1 or iter2 <= len2:
        # assert iter1 <= len1 or iter2 <= len2
        if c1 == c2:
            # print('==')
            old_index = index
            old_iter1 = iter1
            old_iter2 = iter2
            while True:
                out_coords[index] = c1
                index += 1
                iter1 += 1
                iter2 += 1
                # print('index')
                # print(index)
                # print('iter1')
                # print(iter1)
                # print('iter2')
                # print(iter2)
                if iter1 >= len1:
                    #n = iter1 - iter2
                    #end_index = old_index + n
                    # print('copying iter1 >= len1')
                    # print('old_index')
                    # print(old_index)
                    # print('old_iter1')
                    # print(old_iter1)
                    out_data[old_index: index] = data1[old_iter1: iter1] + data2[old_iter2: iter2]
                    #index = end_index
                    break
                if iter2 >= len2:
                    #n = iter1 - iter2
                    #end_index = old_index + n
                    # print('copying iter2 >= len2')
                    # print('old_index')
                    # print(old_index)
                    # print('old_iter2')
                    # print(old_iter2)
                    out_data[old_index: index] = data1[old_iter1: iter1] + data2[old_iter2: iter2]
                    #index = end_index
                    break
                c1 = coords1[iter1]
                c2 = coords2[iter2]
                if c1 != c2:
                    #n = iter1 - iter2
                    #end_index = old_index + n
                    out_data[old_index: index] = data1[old_iter1: iter1] + data2[old_iter2: iter2]
                    #index = end_index
                    break
        if c1 < c2:
            # print('<<')
            old_index = index
            old_iter1 = iter1
            while True:
                out_coords[index] = coords1[iter1]
                index += 1
                iter1 += 1
                # print('index')
                # print(index)
                # print('iter1')
                # print(iter1)
                if iter1 >= len1:
                    #n = iter1 - old_iter1
                    #end_index = old_index + n
                    # print('copying iter1 >= len1')
                    # print('old_index')
                    # print(old_index)
                    # print('old_iter1')
                    # print(old_iter1)
                    out_data[old_index: index] = data1[old_iter1: iter1]
                    # print('copied')
                    #index = end_index
                    break
                c1 = coords1[iter1]
                if c1 >= c2:
                    #n = iter1 - old_iter1
                    #end_index = old_index + n
                    out_data[old_index: index] = data1[old_iter1: iter1]
                    #index = end_index
                    break
        if c2 < c1:  
            # print('>>')          
            old_index = index
            old_iter2 = iter2
            while True:
                out_coords[index] = coords2[iter2]
                index += 1
                iter2 += 1
                # print('index')
                # print(index)
                # print('iter2')
                # print(iter2)
                if iter2 >= len2:
                    #n = iter2 - old_iter2
                    #end_index = old_index + n
                    # print('copying iter2 >= len2')
                    # print('old_index')
                    # print(old_index)
                    # print('old_iter2')
                    # print(old_iter2)
                    out_data[old_index: index] = data2[old_iter2: iter2]
                    # print('copied')
                    #index = end_index
                    break
                c2 = coords2[iter2]
                if c2 >= c1:
                    #n = iter2 - old_iter2
                    #end_index = old_index + n
                    out_data[old_index: index] = data2[old_iter2: iter2]
                    #index = end_index
                    break
        
        if iter1 >= len1 and iter2 < len2:
            # print('1')
            # print('iter1')
            # print(iter1)
            # print('iter2')
            # print(iter2)
            # print('len1')
            # print(len1)
            # print('len2')
            # print(len2)
            end_index = index + len2 - iter2
            out_coords[index: end_index] = coords2[iter2:]
            out_data[index: end_index] = data2[iter2:]
            return ret_out_coords[:, :end_index], out_data[:end_index]
            #break
        
        elif iter2 >= len2 and iter1 < len1:
            # print('2')
            end_index = index + len1 - iter1
            out_coords[index: end_index] = coords1[iter1:]
            out_data[index: end_index] = data1[iter1:]
            return ret_out_coords[:, :end_index], out_data[:end_index]
            #break
        elif iter1 >= len1 and iter2 >= len2:
            # print('3')
            return ret_out_coords[:, :index], out_data[:index]
            #break
        # print('4')

    
    return ret_out_coords[:, :end_index], out_data[:end_index]
            

#@profile
#jit(nogil=True, nopython=False, force_obj=True, cache=True, error_model='numpy')
#jit(nogil=True, nopython=True, cache=True)
def _sum3(coords1, data1, coords2, data2):
    iter1 = 0
    iter2 = 0
    #index = 0
    len1 = coords1.shape[0]
    len2 = coords2.shape[0]
    alloc_length = len1 + len2
    ret_out_coords = np.empty((1, alloc_length), dtype=np.int64)
    out_coords = ret_out_coords[0]
    out_data = np.empty(alloc_length)
    #while not (iter1 == len1 and iter2 == len2):
    #while iter1 < len1 and iter2 < len2:
    c1 = coords1[0]
    c2 = coords2[0]
    index = 0
    while iter1 < len1 or iter2 < len2:
        if iter1 < len1 and iter2 < len2:
            if c1 == c2:
                out_coords[index] = c1
                out_data[index] = data2[iter2] + data1[iter1]
                iter1 += 1
                if iter1 >= len1:
                    break
                iter2 +=1
                if iter2 >= len2:
                    break
                c1 = coords1[iter1]
                c2 = coords2[iter2]
                index += 1
            elif c1 < c2:
                old_iter = iter1
                while True:
                    iter1 += 1
                    if iter1 >= len1:
                        break
                    c1 = coords1[iter1]
                    if c1 >= c2:
                        break
                n = iter1 - old_iter
                end_index = index + n
                out_coords[index: end_index] = coords1[old_iter:iter1]
                out_data[index: end_index] = data1[old_iter: iter1]
                index = end_index
                if iter1 >= len1:
                    break
            else:
                old_iter = iter2
                while True:
                    iter2 += 1
                    if iter2 >= len2:
                        break
                    c2 = coords2[iter2]
                    if c2 >= c1:
                        break
                n = iter2 - old_iter
                end_index = index + n
                out_coords[index: end_index] = coords2[old_iter: iter2]
                out_data[index: end_index] = data2[old_iter: iter2]
                index = end_index
                if iter2 >= len2:
                    break
        elif iter1 < len1:
            end_index = index + len1 - iter1
            out_coords[index: end_index] = coords1[iter1:]
            out_data[index: end_index] = data1[iter1:]
            index = end_index
            break
        #elif iter2 < len2:
        else:
            end_index = index + len2 - iter2
            out_coords[index: end_index] = coords2[iter2:]
            out_data[index: end_index] = data2[iter2:]
            index = end_index
            break
    out_data = out_data[:index]
    return ret_out_coords[:, :index], out_data


#@profile
@jit(nogil=True, nopython=True, cache=True, fastmath=True)
def _sum2(coords1, data1, coords2, data2):
    iter = 0
    len1 = coords1.shape[0]
    len2 = coords2.shape[0]
    ret_out_coords = np.empty((1, len1 + len2), dtype=np.int64)
    out_coords = ret_out_coords[0]
    out_data = np.empty(len1 + len2)
    if coords1[-1] <= coords2[-1]:
        finish_first_coords = coords1
        finish_last_coords = coords2
        finish_first_data = data1
        finish_last_data = data2
        finish_last_len = len2
        finish_first_len = len1
    else:
        finish_first_coords = coords2
        finish_last_coords = coords1
        finish_first_data = data2
        finish_last_data = data1
        finish_last_len = len1
        finish_first_len = len2
    
    i = 0
    index = -1
    c2 = finish_last_coords[0]
    c1 = finish_first_coords[0]
    while True:
        #c2 = finish_last_coords[iter]
        index += 1
        if c1 == c2:
            out_coords[index] = c1
            out_data[index] = finish_first_data[i] + finish_last_data[iter]
            if i == finish_first_len - 1:
                break
            i += 1
            c1 = finish_first_coords[i]
            iter += 1
            c2 = finish_last_coords[iter]
        elif c1 < c2:
            out_coords[index] = c1
            out_data[index] = finish_first_data[i]
            if i == finish_first_len - 1:
                break
            i += 1
            c1 = finish_first_coords[i]
        else:
            out_coords[index] = c2
            out_data[index] = finish_last_data[iter]
            iter += 1
            c2 = finish_last_coords[iter]
    if iter < finish_last_len:
        index += 1
        end_index = index + finish_last_len - iter
        out_coords[index: end_index] = finish_last_coords[iter:]
        out_data[index: end_index] = finish_last_data[iter:]
    return ret_out_coords[:, :end_index], out_data[:end_index]

    



SparseExtensionArray._add_arithmetic_ops()
SparseExtensionArray._add_comparison_ops()
registry.register(SparseArrayType)


#@jit(void(float64[:], int64[:], int64, float64), parallel=True, nogil=True, nopython=True)
@jit(cache=True, nogil=True, nopython=True)
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

module = sys.modules[__name__]
_dtypes = {}
for dtype in ['int8', 'int16', 'int32', 'int64', 'float32', 'float64', 'bool_', 'bool',
              'uint8', 'uint16', 'uint32', 'uint64']:

    if dtype == 'bool' or dtype == 'bool_':
        name = 'bool'
        np_type = getattr(np, 'bool_')
    elif dtype.startswith('u'):
        name = "U{}".format(dtype[1:].capitalize())
        np_type = getattr(np, dtype)
    else:
        name = dtype.capitalize()
        np_type = getattr(np, dtype)
    classname = "{}Dtype".format(name)
    attributes_dict = {'type': np_type,
                        'name': name}
    dtype_type = type(classname, (SparseArrayType, ), attributes_dict)
    setattr(module, classname, dtype_type)
    # register
    registry.register(dtype_type)
    _dtypes[dtype] = dtype_type()