from pandas.core.arrays import ExtensionArray
from collections import Iterable
from pandas.api.extensions import ExtensionDtype, take
import sparse
import operator
import numpy as np

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


class SparseExtensionArray(ExtensionArray):
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
    
    def __setitem__(self, key, value):
        # TODO if value is self.na_value, remove that index from coords and data
        coords_shape = self.data.coords.shape
        coords = self.data.coords.ravel()
        if isinstance(key, int):
            if key not in self.data.coords:
                self.data.coords = np.append(self.data.coords, key).reshape(coords_shape[0], self.data.coords.shape[1] + (len(key) if isinstance(key, Iterable) else 1)) 
                self.data.data = np.append(self.data.data, value)
                if key >= self.data.shape[0]:
                    self.data.shape = (self.data.shape[0] + 1,)
            else:
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
                    self.data.coords = np.append(self.data.coords, i).reshape(coords_shape[0], self.data.coords.shape[1] + 1)
                    self.data.data = np.append(self.data.data, value)
                    if i >= self.data.shape[0]:
                        self.data.shape = (self.data.shape[0] + 1,)
                else:
                    idx = np.where(coords == i)[0][0]
                    self.data.data[idx] = value
        elif isinstance(key, (np.ndarray)):
            if isinstance(key[0], np.bool_):
                for ind, v in enumerate(key):
                    if v:
                        if ind not in coords:
                            self.data.coords = np.append(self.data.coords, ind).reshape(coords_shape[0], self.data.coords.shape[1] + 1) 
                            self.data.data = np.append(self.data.data, value)
                            if ind >= self.data.shape[0]:
                                self.data.shape = (self.data.shape[0] + 1,)
                        else:
                            idx = np.where(coords == i)[0][0]
                            self.data.data[idx] = value
            else:
                for k in key:
                    ind = k if k >= 0 else self.shape[0] + k
                    if ind not in coords:
                        self.data.coords = np.append(self.data.coords, ind).reshape(coords_shape[0], self.data.coords.shape[1] + 1) 
                        self.data.data = np.append(self.data.data, value)
                        if ind >= self.data.shape[0]:
                            self.data.shape = (self.data.shape[0] + 1,)
                    else:
                        idx = np.where(coords == ind)[0][0]
                        self.data.data[idx] = value

        #else:
        #    self.data.data[key] = value


    def __iter__(self):
        # Do we want to iterate through the full blown array or just the non-na values?
        # something like [list(data[:3])] + [np.nan] can be handled better than this
        return iter(self.data.todense().tolist())
 


    @property
    def dtype(self):
        return self._dtype
    
    @classmethod
    def _from_sequence(cls, scalars):
        return SparseExtensionArray(np.array(scalars, dtype=float))
    
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
        return self.data.shape[0]
    
    def __getitem__(self, item):
        coords = self.data.coords.ravel()
        if isinstance(item, int):
            if item < 0:
                item = self.shape[0] + item
            if item < self.shape[0] and item not in self.data.coords:
                return self.na_value
                #return np.nan
            data_row = np.where(coords == item)[0][0]
            return (self.data.data[data_row])
        elif isinstance(item, slice):
            slice_start = item.start
            slice_stop = item.stop
            slice_step = item.step
            if slice_start and slice_start < 0:
                slice_start = self.data.shape[0] + slice_start
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
        elif isinstance(item, np.ndarray):
            values = []
            if isinstance(item[0], np.bool_):
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
            mask = (indices == -1)
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
                #took[i] = fill_value
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


    


    