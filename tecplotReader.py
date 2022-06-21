#!/usr/bin/env python3
#%%
import construct
import numpy as np
import logging
import struct
import tempfile
import os
import shlex
from enum import IntEnum
import shutil
from subprocess import PIPE, Popen
from collections import Counter, OrderedDict
from itertools import tee, count

# Signed little endian
from construct import Int32sl as Int
from construct import Int32sl as Chr
from construct import Float32l as Float
from construct import Float64l as Float64
from construct import Int8ul as UInt8

logging.basicConfig(format='%(levelname)s %(threadName)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARN) # change this line if you don't want verbose info

# global read-only variable
PREPLOT = shutil.which('preplot')
EOHMARKER = 357.0
UNSUPPORTED_MARKER = {
    'Geom': 399.0,
    'Text': 499.0,
    'CustomLabel': 599.0,
    'UserRec': 699.0,
    'DataSetAux': 799.0,
    'VarAux': 899.0
}


RESERVED_KEY = '__internal__'
INTERNAL_KEY = ['_VarDtype_',
            '_PassiveVars_',
            '_PassiveVarDict_',
            '_VarSharing_',
            '_ShareVarDict_',
            '_ConnSharing_',
            'MinVals',
            'MaxVals',
            '_Connect_',
            '_StartByte_',
            '_EndByte_',
            '_Header_']

INT_SIZE = Int.sizeof()  # size of int32 in byte
FLOAT_SIZE = Float.sizeof()

class ZoneType(IntEnum):
    ORDERED = 0
    FELINESEG = 1
    FETRIANGLE = 2
    FEQUADRILATERAL = 3
    FETETRAHEDRON = 4
    FEBRICK = 5
    FEPOLYGON = 6
    FEPOLYHEDRON = 7

FEMNumNode = {
    ZoneType.FELINESEG: 2,
    ZoneType.FETRIANGLE: 3,
    ZoneType.FEQUADRILATERAL: 4,
    ZoneType.FEBRICK: 8,
    ZoneType.FETETRAHEDRON: 4
}

class VarLocType(IntEnum):
    NODAL = 0
    CELLCENTERED = 1

def uniquify(seq, suffs = count(1)):
    """Make all the items unique by adding a suffix (1, 2, etc).

    `seq` is mutable sequence of strings.
    `suffs` is an optional alternative suffix iterable.
    """
    not_unique = [k for k,v in Counter(seq).items() if v>1]
    if len(not_unique) > 0:
        logger.warning('Variable name: "{:s}" is not unique'.format(', '.join(not_unique)))
    # suffix generator dict - e.g., {'name': <my_gen>, 'zip': <my_gen>}
    suff_gens = dict(zip(not_unique, tee(suffs, len(not_unique))))
    for idx,s in enumerate(seq):
        try:
            suffix = str(next(suff_gens[s]))
        except KeyError:
            # s was unique
            continue
        else:
            seq[idx] += suffix


class BinArray:
    def __init__(self, count, dtype):
        self.count = count
        self.dtype = dtype
        self.__stride__ = self.dtype.sizeof()

    def parse(self, byte_list):
        offset = 0
        r = []
        for i in range(self.count):
            r.append(self.dtype.parse(byte_list[offset:offset+self.__stride__]))
            offset += self.__stride__
        return r

    def sizeof(self):
        return self.__stride__ * self.count


def parse_str(byte_list, offset = 0, dtype=Int):
    """
    Parse a null terminated string from byte_list

    Argument
    ---------
    `byte_list` byte string
    `offset`    offset of parsing

    Return
    ---------
        str
        new_offset
    """
    r = ''
    new_offset = offset
    while offset < len(byte_list):
        c, offset = parse_buffer(byte_list, dtype, offset)
        r += chr(c)
        new_offset += dtype.sizeof()
        if c == 0:
            break
    return r[:-1], new_offset


def parse_buffer(byte_list, dtype, offset=0, count=1):
    """
    Parse byte_list

    `dtype` data type. It should have a method `parse`
    `count` number of this type to parse
    `offset` start of the parsing
    `byte_list` python binary array or list like object

    """
    if dtype == str:
        return parse_str(byte_list, offset)
    else:
        if type(dtype) == construct.core.Array:
            raise TypeError('dtype should be plain data type')
        if count != 1:
            dtype = construct.Array(count, dtype)
        new_offset = offset + dtype.sizeof()
        s = dtype.parse(byte_list[offset:new_offset])
        if count != 1:
            s = list(s)
        return s, new_offset


def parse_schema(byte_list, dtype, offset = 0):
    if type(dtype) == list:
        r = []
        for idtype in dtype:
            rr, offset = parse_schema(byte_list, idtype, offset)
            r.append(rr)
        return r, offset
    elif isinstance(dtype, dict):
        r = OrderedDict()
        for k, vdtype in dtype.items():
            rr, offset = parse_schema(byte_list, vdtype, offset)
            r[k] = rr
        return r, offset
    else:
        return parse_buffer(byte_list, dtype, offset)


class Struct:
    """
    Helper class to convert dict to a class and make autocomplete possible in interactive run

    """
    def __init__(self, **entries):
        for key in entries.keys():
            if type(entries[key]) == dict:
                entries[key] = Struct(**entries[key])
            elif type(entries[key]) == list:
                val = []
                for i in entries[key]:
                    if type(i) == dict:
                        val.append(Struct(**i))
                    else:
                        val.append(i)
                entries[key] = val

        self.__dict__.update(entries)

    def __getitem__(self, key):
        if key in self.__dict__:
            return self.__dict__[key]
        else:
            raise KeyError(key)

    def __setitem__(self, key, val):
        self.__dict__[key] = val

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def __iter__(self):
        return iter(self.__dict__)

    def __repr__(self):
        return self.__dict__.__repr__()

    def values(self):
        return self.__dict__.values()


class ZoneDataOrderedDict(OrderedDict):
    def __init__(self, *args, **kwargs):
        self[RESERVED_KEY] = OrderedDict({i: None for i in INTERNAL_KEY})
        super().__init__(*args, *kwargs)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)

    def __call__(self, ivar):
        '''
        Get the data based on index of the variable
        '''
        k = list(self.keys())[ivar+1]
        return self[k]

class BinaryFile:
    '''
    Context manager with getitem
    '''
    def __init__(self, *args, **kwargs):
        if len(args) == 1:
            args = (args[0], 'rb')
        self.f = open(*args, **kwargs)
        self.size = os.path.getsize(args[0]) # size of file in byte

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __getitem__(self, index):
        if type(index) == slice:
            if index.step:
                raise ValueError(index.step)
            start = index.start or 0
            self.seek(start, os.SEEK_SET)
            if index.stop is None:
                return self.read()
            else:
                return self.read(index.stop-start)
        else:
            self.seek(index, os.SEEK_SET)
            return self.read(1)

    def read(self, *args, **kwargs):
        return self.f.read(*args, **kwargs)

    def close(self):
        return self.f.close()

    def seek(self, *args, **kwargs):
        return self.f.seek(*args, **kwargs)

    def __len__(self):
        return self.size


def genTempFilePath(subdir=''):
    tempFileName = next(tempfile._get_candidate_names())
    tempDir = tempfile._get_default_tempdir()
    if subdir == '':
        return os.path.join(tempDir, tempFileName)
    else:
        dir = os.path.join(tempDir, subdir)
        os.makedirs(dir, exist_ok=True)
        return os.path.join(dir, tempFileName)


def isTecBinary(file):
    with open(file, 'rb') as f:
        fc = f.read(8)
    magicKey = [i.decode('ascii') for i in struct.unpack('c'*8, fc)]
    if ''.join(magicKey) == '#!TDV112':
        return True
    elif ''.join(magicKey) == '#!TDV191':
        logger.warn('#!TDV191 format is not fully supportted')
        return True
    else:
        return False


def convertAscii2Bin(file, saveFile=False):
    if not PREPLOT:
        raise ValueError('Preplot is not in PATH, we need it to convert ASCII to binary')
    name = os.path.basename(file)
    if saveFile:
        newName = name.split('.')[0] + '_bin.plt'
        newFile = os.path.join(os.path.dirname(file), newName)
    else:
        newFile = genTempFilePath() + ".plt"
    command = f'"{PREPLOT}" "{file}" "{newFile}"'
    print(command)
    c = Popen(shlex.split(command), stdout=PIPE, stderr=PIPE, cwd=os.getcwd())
    o, e = c.communicate()
    o = o.decode('utf8')
    e = e.decode('utf8')
    if len(e) > 0:
        raise ValueError('Error: {:e}'.format(e))
    return newFile

#%%
def read_zone_record_single(byte_list, offset, variables):
    start_byte = offset
    title, offset = parse_str(byte_list, offset)

    __parent_zone__, offset = parse_buffer(byte_list, Int, offset)
    # assert _parent_zone == -1

    strand_id, offset = parse_buffer(byte_list, Int, offset)
    # if strand_id == - 1:
    #     logger.debug('StandID is unset')

    solution_time, offset = parse_buffer(byte_list, Float64, offset)

    __default_zone_color__, offset = parse_buffer(byte_list, Int, offset)

    zone_type, offset = parse_buffer(byte_list, Int, offset)
    zone_type = ZoneType(zone_type)

    var_loc, offset = parse_buffer(byte_list, Int, offset)
    var_loc = OrderedDict({v: var_loc for v in variables})
    if var_loc[variables[0]] == 1:
        for i, v in enumerate(variables):
            var_loc[v], offset = parse_buffer(byte_list, Int, offset)
    var_loc = OrderedDict({k: VarLocType(v) for k, v in var_loc.items()})

    raw_face_neighbors, offset = parse_buffer(byte_list, Int, offset)

    num_user_defined_face_neighbors, offset = parse_buffer(byte_list, Int, offset)

    user_defind_face_neighbor_mode = -1
    miscellaneous_face_neighbor = -1
    if num_user_defined_face_neighbors != 0:
        user_defind_face_neighbor_mode, offset = parse_buffer(byte_list, Int, offset)
        if zone_type != ZoneType.ORDERED:
            miscellaneous_face_neighbor, offset = parse_buffer(byte_list, Int, offset)

    zone_dim = OrderedDict()
    if zone_type == ZoneType.ORDERED:
        size, offset = parse_buffer(byte_list, Int, offset, 3)
        zone_dim = OrderedDict({
            'I': size[0],
            'J': size[1],
            'K': size[2]
        })
    else:
        num_pts, offset = parse_buffer(byte_list, Int, offset)
        if zone_type in [ZoneType.FEPOLYGON, ZoneType.FEPOLYHEDRON]:
            raise ValueError('Unsupported FEM type')
        num_elems, offset = parse_buffer(byte_list, Int, offset)
        tmp, offset = parse_buffer(byte_list, Int, offset, 3)
        zone_dim = OrderedDict({
            'Nodes': num_pts,
            'Elements': num_elems,
            '__ICellDim__': tmp[0],
            '__JCellDim__': tmp[1],
            '__KCellDim__': tmp[2]
        })

    has_aux_data, offset = parse_buffer(byte_list, Int, offset)
    aux_data = []
    while has_aux_data == 1:
        __aux_name, offset = parse_str(byte_list, offset)
        __aux_type, offset = parse_buffer(byte_list, Int, offset)
        assert __aux_type == 0
        __aux_val, offset = parse_str(byte_list, offset)
        aux_data.append((__aux_name, __aux_val))
        has_aux_data, offset = parse_buffer(byte_list, Int, offset)

    end_byte = offset
    __rdict = OrderedDict({
        "title": title,
        "__parent_zone__": __parent_zone__,
        "strand_id": strand_id,
        "solution_time": solution_time,
        "__default_zone_color__": __default_zone_color__,
        "zone_type": zone_type,
        "var_loc": var_loc,
        "raw_face_neighbors": raw_face_neighbors,
        "num_user_defined_face_neighbors": num_user_defined_face_neighbors,
        "user_defind_face_neighbor_mode": user_defind_face_neighbor_mode,
        "miscellaneous_face_neighbor": miscellaneous_face_neighbor,
        "__zone_dim__": zone_dim,
        "aux_data": aux_data,
        "__start_byte__": start_byte,
        "__end_byte__": end_byte
    })

    # the __start_byte__ / __end_byte__ here should be the same as the one obtained by
    # find_zone_header
    __rdict.update(zone_dim)

    return __rdict

#%%
def find_zone_header(byte_list, offset=-1):
    if offset < 0:
        file_header = read_file_header(byte_list)
    offset = file_header['__end_byte__']

    zone_version = []
    zone_start_markers = []  # end of the 299.0 or 298.0 magic key
    zone_end_markers = [] # start of the 299.0, 298.0, or other unsupported markeres
    while offset < len(byte_list):
        eof_value, offset = parse_buffer(byte_list, Float, offset)
        if eof_value == 299.0:
            zone_start_markers.append(offset)
            zone_end_markers.append(offset - FLOAT_SIZE)
            zone_version.append(112)
        elif eof_value == 298.0:
            zone_start_markers.append(offset)
            zone_end_markers.append(offset - FLOAT_SIZE)
            zone_version.append(191)
        elif eof_value in list(UNSUPPORTED_MARKER.values()) + [EOHMARKER]:
            zone_end_markers.append(offset - FLOAT_SIZE)
            break

    zone_markers = [
        (zone_start_markers[i], zone_end_markers[i+1])
        for i in range(len(zone_start_markers))
    ]
    return zone_markers, zone_version


def find_end_of_header(byte_list, offset = -1):
    if offset < 0:
        file_header = read_file_header(byte_list)
        zone_header = read_zone_header(byte_list, file_header)
        offset = zone_header[-1]['__end_byte__']

    while offset < len(byte_list):
        eof_value, offset = parse_buffer(byte_list, Float, offset)
        if eof_value == EOHMARKER:
            break
    return offset - FLOAT_SIZE


def find_zones_data(byte_list, end_of_header = -1):
    if end_of_header < 0:
        end_of_header = find_end_of_header(byte_list)

    offset = end_of_header
    while offset < len(byte_list):
        data_marker, offset = parse_buffer(byte_list, Float, offset)
        if data_marker == 299.0:
            break
    return offset-FLOAT_SIZE

#%%
def read_file_header(byte_list):
    start_byte = 0
    __magic_num, offset = parse_buffer(byte_list, UInt8, start_byte, 8)
    __magic_num = ''.join([chr(i) for i in __magic_num])
    if __magic_num[:5] != '#!TDV':
        raise ValueError('Wrong file type')
    version = int(__magic_num[5:])

    __byte_order, offset = parse_buffer(byte_list, Int, offset)
    if __byte_order != 1:
        raise ValueError('Wrong data type for INT32')

    __file_type_name=['FULL','GRID','SOLUTION']
    filetype, offset = parse_buffer(byte_list, Int, offset)
    filetype = __file_type_name[filetype]

    title, offset = parse_str(byte_list, offset)

    __num_vars, offset = parse_buffer(byte_list, Int, offset)

    variables  = [f'V{i+1}' for i in range(__num_vars)]
    for i in range(__num_vars):
        variables[i], offset = parse_str(byte_list, offset)
    uniquify(variables)

    # It's strange that in the manual, the block after file header should be zone record
    # But the file generated by preplot doesn't seem like it
    aux_data = []
    tmp, new_offset = parse_buffer(byte_list, Float, offset)
    while tmp != 299.0 and tmp != 298.0:
        if tmp == 799.0:
            offset = new_offset
            aux_var_name, offset = parse_str(byte_list, offset)
            __aux_var_type, offset = parse_buffer(byte_list, Int, offset)
            aux_var_val, offset = parse_str(byte_list, offset)
            aux_data.append((aux_var_name, aux_var_val))
        elif tmp in UNSUPPORTED_MARKER.values():
            offset = new_offset
        elif tmp == EOHMARKER or new_offset == len(byte_list):
            raise ValueError('No zone record is found')
        else:
            offset = new_offset

        assert offset+FLOAT_SIZE != len(byte_list)
        tmp, new_offset = parse_buffer(byte_list, Float, offset)


    end_byte = new_offset - FLOAT_SIZE

    return OrderedDict({
        '__version__' : version,
        'filetype'  : filetype,
        'title':title,
        'variables':variables,
        '__start_byte__': start_byte,
        '__end_byte__': end_byte,
        '__end_of_header__': -1
    })

    # byte_list[end_of_header: end_of_header+4] should be 357.0
    # __end_of_header__ = find_end_of_header(byte_list, offset)
    # __end_of_file_header__ = offset

    # zone_data_marker = find_zones_data(byte_list, __end_of_header__)

    # return OrderedDict({
    #     '__version__' : version,
    #     'filetype'  : filetype,
    #     'title':title,
    #     'variables':variables,
    #     '__end_of_header__': __end_of_header__,
    #     '__end_of_file_header__': __end_of_file_header__,
    #     '__start_of_data__': zone_data_marker
    #     })

#%%
def read_zone_header(byte_list, file_header=None):
    if file_header is None:
        file_header = read_file_header(byte_list)
    offset = file_header['__end_byte__']

    zone_header = []
    while offset < len(byte_list):
        # previously, I use find_zone_header
        # But it is very inefficient for datafile with manu manu zones
        tmp, offset = parse_buffer(byte_list, Float, offset)
        if tmp == 299.0:
            zone_version = 112
        elif tmp == 298.0:
            zone_version = 191
        else:
            break
        zone_header.append(read_zone_record_single(byte_list, offset, file_header['variables']))
        offset = zone_header[-1]['__end_byte__']
        zone_header[-1]['version'] = zone_version
        tmp, new_offset = parse_buffer(byte_list, Float, offset)
        if tmp in list(UNSUPPORTED_MARKER.values()) + [EOHMARKER]:
            break
    return zone_header


#%%
def read_zone_data(byte_list, zone_markers, file_header, zone_header, zone_counter=0):
    '''
    Given the byte_list, read zones starting from zone_markers

    `byte_list`    COMPLETE byte list
    `file_header`  file header
    `zone_markers` start byte of the zones to load
    `zone_counter` index of the first zone to load, its header should be zone_header[zone_counter]
    '''
    var_names = file_header['variables']
    zones_list=[]

    for start_byte in zone_markers:
        offset = start_byte
        zone_data = ZoneDataOrderedDict()
        internal_data = zone_data[RESERVED_KEY]

        magicByte, offset = parse_buffer(byte_list, Float, offset)
        if magicByte!= 299.0:
            raise ValueError('Wrong data starting byte')

        var_dtype, offset = parse_buffer(byte_list, Int, offset, len(var_names))
        var_dtype = OrderedDict(zip(var_names, var_dtype))
        np_dtype = [None, np.float32, np.float64, np.int_, np.short, np.byte, np.int8]
        for k in var_names:
            var_dtype[k] = np_dtype[var_dtype[k]]
        internal_data['_VarDtype_'] = var_dtype

        internal_data['_PassiveVars_'], offset = parse_buffer(byte_list, Int, offset)
        # 0 : non passive 1: passive
        if internal_data['_PassiveVars_']  != 0:
            passive_var_dict, offset = parse_buffer(byte_list, Int, offset, len(var_names))
            internal_data['_PassiveVarDict_'] = OrderedDict(zip(var_names, passive_var_dict))
        else:
            internal_data['_PassiveVarDict_'] = OrderedDict({i: 0 for i in var_names})

        internal_data['_VarSharing_'], offset = parse_buffer(byte_list, Int, offset)
        # -1: non sharing >=0: index of var to share with
        if internal_data['_VarSharing_']  != 0:
            share_var_dict, offset = parse_buffer(byte_list, Int, offset, len(var_names))
            internal_data['_ShareVarDict_'] = OrderedDict(zip(var_names, share_var_dict))
        else:
            internal_data['_ShareVarDict_'] = OrderedDict({i: -1 for i in var_names})

        internal_data['_ConnSharing_'], offset = parse_buffer(byte_list, Int, offset)

        non_passive_non_shared = []
        if internal_data['_VarSharing_'] != 0:
            non_passive_non_shared = [i for i, v in internal_data['_ShareVarDict_'].items() if v == -1]
        else:
            non_passive_non_shared = var_names[:]

        if internal_data['_PassiveVars_'] != 0:
            for name in var_names:
                if internal_data['_PassiveVarDict_'][name] != 0:
                    if name in non_passive_non_shared:
                        non_passive_non_shared.remove(name)

        min_val = OrderedDict({v: 0 for v in var_names})
        max_val = OrderedDict({v: 0 for v in var_names})
        for var_with_min_max in non_passive_non_shared:
            min_val[var_with_min_max], offset = parse_buffer(byte_list, Float64, offset)
            max_val[var_with_min_max], offset = parse_buffer(byte_list, Float64, offset)

        for v in var_names:
            if v not in min_val.keys():
                if internal_data['_ShareVarDict_'][v] != -1:
                    vv = var_names[internal_data['_ShareVarDict_'][v]]
                    min_val[v] = min_val[vv]
                    max_val[v] = max_val[vv]

        internal_data['MinVals'] = min_val
        internal_data['MaxVals'] = max_val

        zt = zone_header[zone_counter]['zone_type']
        zh = zone_header[zone_counter]
        nmaxlenvar = max([len(i) for i in non_passive_non_shared])
        for ivar, name in enumerate(non_passive_non_shared):
            varloc = zh['var_loc'][name]
            shape = []
            if zt == ZoneType.ORDERED:
                Imax = zh['I']
                Jmax = zh['J']
                Kmax = zh['K']
                if varloc == 0:
                    ndata = Imax * Jmax * Kmax
                    shape.append(Imax)
                    if Jmax > 1:
                        shape.append(Jmax)
                    if Kmax > 1:
                        shape.append(Kmax)
                else:
                    ndata = Imax
                    shape.append(Imax)
                    if Jmax > 1:
                        ndata *= Jmax
                        shape.append(Jmax)
                    if Kmax > 1:
                        ndata *= Kmax - 1
                        shape.append(Kmax-1)
            else:
                NumPts = zh['Nodes']
                NumElements = zh['Elements']
                if varloc == 0:
                    ndata = NumPts
                else:
                    ndata = NumElements

            if isinstance(byte_list, bytes):
                data = np.frombuffer(byte_list, dtype=internal_data['_VarDtype_'][name],
                                count=ndata,
                                offset=offset)
            elif isinstance(byte_list, BinaryFile):
                byte_list.seek(offset, os.SEEK_SET)
                data = np.fromfile(byte_list.f, dtype=internal_data['_VarDtype_'][name],
                                count=ndata)
                #logger.info(f"Finish loading Zone {zone_counter+1} Variable {name:<{nmaxlenvar+1}s} started at {offset} byte")
            offset = offset + data.dtype.itemsize * data.size
            end_byte = offset

            if zt == ZoneType.ORDERED:
                data = np.reshape(data, shape, order='F')
                if varloc == 1:
                    data = data[:-1, :-1,:]
            zone_data[name] = data


        if zt != ZoneType.ORDERED:
            if zt not in FEMNumNode.keys():
                raise ValueError('Unsupported FEM type')
            NumPts = zh['Nodes']
            NumElements = zh['Elements']
            connect = np.frombuffer(byte_list, dtype=internal_data['_VarDtype_'][name],
                            count=NumElements * FEMNumNode[zt],
                            offset=offset)
            offset = offset + NumElements * FEMNumNode[zt]
            end_byte = offset
            internal_data['_Connect_'] = connect

        internal_data['_StartByte_'] = start_byte
        internal_data['_EndByte_'] = end_byte
        internal_data['_Header_'] = zh
        zones_list.append(zone_data)

        zone_counter = zone_counter + 1

    return zones_list, zone_counter

#%%
def read_all_data(byte_list, file_header=None, zone_header=None, start_byte=-1):
    '''
    byte_list:
        - option1: a list of byte
        - option2: a BinaryFile class
    '''
    if file_header == None:
        file_header = read_file_header(byte_list)
    if zone_header == None:
        zone_header = read_zone_header(byte_list, file_header)
    if start_byte < 0:
        end_of_header_byte  = file_header['__end_of_header__']
        if end_of_header_byte == -1 or Float.parse(byte_list[end_of_header_byte:end_of_header_byte+4]):
            end_of_header_byte = find_end_of_header(byte_list, zone_header[-1]['__end_byte__'])
        start_byte = end_of_header_byte + FLOAT_SIZE

    zones_list = []
    zone_counter = 0
    for izone in range(len(zone_header)):
        # read zone one by one
        r, zone_counter = read_zone_data(byte_list, [start_byte], file_header, zone_header, zone_counter)
        start_byte = r[0][RESERVED_KEY]['_EndByte_']
        zones_list.append(r[0])

    # Handle VarSharing
    for izone in range(len(zone_header)):
        if zones_list[izone][RESERVED_KEY]['_VarSharing_'] == 1:
            for k, i in zones_list[izone][RESERVED_KEY]['_ShareVarDict_'].items():
                if i >= 0:
                    zones_list[izone][k] = zones_list[i][k]
    return zones_list

#%%
MAX_READ = 1024 * 1024 # 1MB
class TecplotFile():
    def __init__(self, filePath):
        if not os.path.isfile(filePath):
            raise FileNotFoundError(filePath)
        self.binary = True
        if isTecBinary(filePath):
            self.binaryFilePath = filePath
        else:
            self.binary = False
            self.binaryFilePath = convertAscii2Bin(filePath, False)
            logger.warning(f'{filePath} is convered to {self.binaryFilePath}')
        self.filePath = filePath

        file_size = os.path.getsize(self.binaryFilePath)
        memory_usage = file_size / 1024 / 1024     # "MB"
        if memory_usage > 1024:
            logger.warning(f'{memory_usage:.2f} MB memory is needed')

        # header is usually small, read the original data
        with BinaryFile(self.binaryFilePath) as f:
            self.header = Struct(**read_file_header(f))
            self.zone_header = [Struct(**i) for i in read_zone_header(f, self.header)]

        self.data = [None] * len(self.zone_header)
        self.dataLoaded = False

        self.variables = self.header.variables

    def __getattribute__(self, item):
        if item == 'data':
            if not self.dataLoaded:
                self.load_data()
        return super().__getattribute__(item)

    def __del__(self):
        try:
            super().__del__(self)
        except AttributeError:
            pass
        if self.filePath != self.binaryFilePath:
            if os.path.isfile(self.binaryFilePath):
                os.remove(self.binaryFilePath)

    def load_data(self, on_demand=True):
        if self.binaryFilePath != self.filePath:
            if not os.path.isfile(self.binaryFilePath):
                convertAscii2Bin(self.filePath, self.binaryFilePath)
        if on_demand:
            # this method consumes less memory since the data is load on the fly
            with BinaryFile(self.binaryFilePath, 'rb') as f:
                self.data = read_all_data(f, self.header, self.zone_header)
        else:
            logger.info("Loading the whole file in memory will take a long time")
            with open(self.binaryFilePath, 'rb') as f:
                self.data = read_all_data(f.read(), self.header, self.zone_header)
        self.dataLoaded = True

        # clean up temporary file
        if self.binaryFilePath != self.filePath:
            if os.path.isfile(self.binaryFilePath):
                os.remove(self.binaryFilePath)


# %%
if __name__ == "__main__":
    file = 'PVT_dense_01_00_crtrs_Nd.dat'
    file = 'efields.dat'
    file = '1500Ws_2000Input_85PulseDC_FullBurst_p054_iedf_01_ion_data_iedf.dat.iadf_1d_v4.plt'
    file = '1500Ws_2000Input_85PulseDC_FullBurst_iedf_01.dat'
    import time

    tic = time.perf_counter()
    tecFile = TecplotFile(file)
    toc = time.perf_counter()

    print(f'Read file and zone header: {toc-tic:.4f} seconds')

    # only header of the file and zones are loaded now
    var_names = tecFile.header.variables
    var_names = tecFile.header['variables']
    header_keys = tecFile.header.keys()

    zone_name = tecFile.zone_header[0].title
    zone_name = tecFile.zone_header[0]['title']  # access as a dict

    # iterate zones' header
    nprint = min(len(tecFile.zone_header), 5)
    for i, iz in enumerate(tecFile.zone_header[:nprint]):
        print(f'Zone {i}')
        for k, v in iz.items():
            print(f'    {k} : {v}')

    # data is loaded when tecFile.data is accessed or tecFile.load_data is called
    tic = time.perf_counter()

    ## you don't need to explicitly invoke the following command
    ## data will be loaded when you access the tecFile.data oject
    # tecFile.load_data(True)
    minVals = tecFile.data[0](-1)['MinVals']
    zone_data = tecFile.data[0] # first zone
    toc = time.perf_counter()
    print(f'Read data: {toc-tic:.4f} seconds')

    zone_data = tecFile.data[0] # first zone

    # access data, the data is a numpy.array
    print(zone_data(0)[:10])            # access data by index of variable name, started from 0
    print(zone_data[var_names[0]][:10]) # access data by variable name


    # use index -1 or RESERVED_KEY to access INTERNAL_KEY
    nmaxlenvar = max(len(v) for v in var_names)
    print('Max value for zone 0:')
    for i, v in enumerate(var_names):
        print(f'   {v:<{nmaxlenvar+2}}:  {zone_data(-1)["MaxVals"][v]}')

    print('Min value for zone 1:')
    for i, v in enumerate(var_names):
        print(f'   {v:<{nmaxlenvar+2}}:  {zone_data(-1)["MinVals"][v]}')


    import signal
    def handler(signum, frame):
        quit()
    signal.signal(signal.SIGINT, handler)
    print("Ctrl + C to exit")
    while True:
        time.sleep(30)
