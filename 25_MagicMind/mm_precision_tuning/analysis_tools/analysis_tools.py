#!/usr/bin/env python

from __future__ import division
import argparse
import getopt
import json
from unittest.loader import VALID_MODULE_NAME
import numpy as np
import os
import sys
import time, re

try:
  from magicmind.tools.debug_tools.proto import tensor_pb2
  from magicmind.tools.debug_tools.utils.common_utils import legalize_char, print_red_color, print_green_color, print_yellow_color
except:
  sys.path.append(os.path.join(os.path.dirname(__file__), "lib/proto"))
  import tensor_pb2

types_pb2=tensor_pb2.DataType

class CompareData(object):

    def __init__(self, src_dir, dst_dir):
        if not os.path.isdir(src_dir) or not os.path.isdir(dst_dir):
            raise ValueError(print_red_color(">>> analysis_tools.py -s <src_dir> -m <dst_dir>   "
                                              "{} and {} should be directory.").format(src_dir, dst_dir))
        self.src_dir = src_dir
        self.dst_dir = dst_dir
        self.src_files = self.get_files(src_dir)
        self.dst_files = self.get_files(dst_dir)
        self.src_files.sort()
        self.dst_files.sort()
        self.record = {}
        self.src_record = {}
        self.dst_record = {}
        self.src_loop_cnt = []
        self.src_info = {}
        self.dst_info = {}
        self.diff = {}
        self.results = {}
        self.alldiffs = []
        self.count_pass = 0

    def get_files(self, dir):
        dir_files = os.listdir(dir)
        return dir_files

class AllFrameworkProcess(object):
    def __init__(self, src_dir, dst_dir, output_dir):
        self.comp_data = CompareData(src_dir, dst_dir)
        self.output_dir = output_dir
        self.json_name = "diff.json"
        os.system("mkdir -vp " + self.output_dir)
        if os.path.exists(self.output_dir + self.json_name):
            os.system("rm " + self.output_dir + self.json_name)

    def process_src_data(self):
        raise NotImplementedError

    def process_dst_data(self):
        raise NotImplementedError
    
    def process_string_name(self):
        raise NotImplementedError

    def preprocess_string(self, file_name):
        has_special_str = re.search(r"[!#$%^&*~]", file_name)
        if has_special_str != None:
            raise ValueError(print_red_color("Your PB name has special string, the name is {}".format(file_name)))

    def dst_preprocess(self):
        # preprocess to get loop count
        self.comp_data.src_loop_cnt = [0 for i in self.comp_data.src_files]

    def src_preprocess(self, src_cnt):
        # judge endswith pb and remove time stamp
        src_file_name = self.comp_data.src_files[src_cnt]
        if not src_file_name.endswith(".pb"):
            raise ValueError(print_red_color("Input file format error, only support .pb file."))
        print(print_green_color("src_tensor:No_%d/total_%d"%(src_cnt,len(self.comp_data.src_files))))
        tensor_name_split = src_file_name.split('_',1)[1]
        return tensor_name_split
    
    def post_process_nomatch(self, src_tensor_name_split):
        diff = {}
        diff["TensorName"] = src_tensor_name_split[:-3]
        diff["Valid"] = False
        self.comp_data.alldiffs.append(diff)

    def post_process_match(self, src_tensor, dst_tensor, src_file_name, dst_file_name, src_trans_key, dst_trans_key):
        diff = {}
        src_info = {}
        dst_info = {}
        results = {}
        src_array = process_data.get_array_from_tensor(src_tensor)
        diff["TensorName"] = src_file_name.split('_',1)[1][:-3]
        src_info["Timestamp"] = src_file_name.split('_',1)[0]
        src_info["NodeName"] = src_tensor.node_name
        src_info["OpType"] = src_tensor.op_type
        src_info["Index"] = src_tensor.index
        diff["TensorShape"] = src_array.shape

        dst_array = process_data.get_array_from_tensor(dst_tensor)
        dst_info["Timestamp"] = dst_file_name.split('_',1)[0]
        dst_info["NodeName"] = dst_tensor.node_name
        dst_info["OpType"] = dst_tensor.op_type
        dst_info["Index"] = dst_tensor.index
        diff1, diff2, diff3, diff4 = process_data.cal_diff(src_file_name.split('_',1)[1][:-3], src_array, dst_array, src_trans_key, dst_trans_key)
        python_type_list = [int, float, bool, complex]
        results["diff1"] = diff1.item() if type(diff1) not in python_type_list else diff1
        results["diff2"] = diff2.item() if type(diff2) not in python_type_list else diff2
        results["diff3"] = np.array(diff3).tolist()
        results["diff4"] = np.array(diff4).tolist()
        diff["SrcModelInfo"] = src_info
        diff["DstModelInfo"] = dst_info
        diff["Valid"] = True
        diff["Results"] = results
        self.comp_data.alldiffs.append(diff)
        
        self.comp_data.count_pass = self.comp_data.count_pass + 1

    def tensor_name_split_layout(self, dst_file_name):
        trans_key = ''
        if re.search("functionalize", str(dst_file_name)) is None:
            dst_file_name_split = str(dst_file_name).split("@axis")
            if len(dst_file_name_split) == 2:
                trans_key = dst_file_name_split[1][:-3]
                if re.match("(-?[0-9])to(-?[0-9])", trans_key):
                    # To process @axis1to-1-1
                    if not re.findall('^(-?[0-9])to(-?\d+)$',trans_key):
                        trans_key = trans_key[:-2]
                elif re.match("\[(\d)?(,\d)*\]", trans_key):
                    # To process @axis[0~8] or @axis[0~8]-1
                    if not re.findall('^\[(\d)?(,\d)*\]$',trans_key):
                        trans_key = trans_key[:-2]
                    trans_check=list(map(int, re.findall("\d+", trans_key)))
                    if len(set(trans_check)) != len(trans_check):
                        raise ValueError(print_red_color("The suffix of this tensor.pb is wrong, suffix = {}".format(trans_key)))
                else:
                    raise ValueError(print_red_color("The suffix of this tensor.pb is wrong, suffix = {}".format(trans_key)))
            elif len(dst_file_name_split) > 2:
                raise ValueError(print_red_color("There are many @axis in pb name, we need 0 or 1 but get {}".format(dst_file_name)))
        return trans_key

    def loop_compare(self):
        for src_cnt, src_tensor_name in enumerate(self.comp_data.src_record.keys()):
            dst_file_name = ''
            dst_trans_key = ''
            src_trans_key = ''
            src_tensor_name_split = self.comp_data.src_record[src_tensor_name]
            if self.comp_data.dst_record.get(src_tensor_name_split) is not None:
                # compare dst tensor and src tensor loop count
                for count_loop in self.comp_data.dst_record[src_tensor_name_split].keys():
                    if count_loop == self.comp_data.src_loop_cnt[src_cnt]:
                        dst_file_name = self.comp_data.dst_record[src_tensor_name_split][count_loop]
                        dst_trans_key = self.tensor_name_split_layout(dst_file_name)
                        break

            if dst_file_name == '':
                self.post_process_nomatch(src_tensor_name_split)
                continue
            src_trans_key = self.tensor_name_split_layout(src_tensor_name)
            src_tensor = tensor_pb2.TensorProto()
            with open(os.path.join(self.comp_data.src_dir, self.comp_data.src_files[src_cnt]), 'rb') as tensor_file:
                src_tensor.ParseFromString(tensor_file.read())
            if src_tensor.version_number != 1:
                print(print_red_color("Version number: " + str(src_tensor.version_number) + " is unsupported yet."))
                return False
            # get dst tensor and info
            dst_tensor = tensor_pb2.TensorProto()
            with open(os.path.join(self.comp_data.dst_dir, dst_file_name), 'rb') as tensor_file:
                dst_tensor.ParseFromString(tensor_file.read())
            if dst_tensor.version_number != 1:
                print(print_red_color("Version number: " + str(dst_tensor.version_number) + " is unsupported yet."))
                return False
            self.post_process_match(src_tensor, dst_tensor, self.comp_data.src_files[src_cnt], dst_file_name, src_trans_key, dst_trans_key)
        print(print_green_color("src tensor match rate:%d/%d"%(self.comp_data.count_pass,len(self.comp_data.src_files))))
        with open(os.path.join(self.output_dir, self.json_name), 'w') as json_file:
            json.dump(self.comp_data.alldiffs, json_file, indent=4, separators=(',', ': '))
        return self.comp_data.count_pass

class TfProcess(AllFrameworkProcess):

    def process_src_data(self):
        num_record = {}
        for i ,src_file in enumerate(self.comp_data.src_files):
            if not src_file.endswith(".pb"):
                raise ValueError(print_red_color("Input file format error, only support .pb file."))
            src_tensor_name_split = self.src_preprocess(i)
            src_tensor_name_split = self.process_string_name(src_tensor_name_split)
            num_record[src_tensor_name_split] = num_record.get(src_tensor_name_split, 0) + 1
            self.comp_data.src_loop_cnt[i] = num_record[src_tensor_name_split]
            self.comp_data.src_record[src_file] = src_tensor_name_split

    def process_dst_data(self):
        for i, dst_file in enumerate(self.comp_data.dst_files):
            if not dst_file.endswith(".pb"):
                raise ValueError(print_red_color("Input file format error, only support .pb file."))
            name = dst_file
            name_split = dst_file.split('_',1)[1]
            dst_tensor = tensor_pb2.TensorProto()
            with open(os.path.join(self.comp_data.dst_dir, name), 'rb') as tensor_file:
                dst_tensor.ParseFromString(tensor_file.read())
                # filt output tensor and name doesn't include cnnl
                if re.search('cnnl',name) is None:
                    name_split = self.process_string_name(name_split)
                    inner_dst = self.comp_data.dst_record.setdefault(name_split,{})
                    # tensor is input
                    if dst_tensor.is_output is not None and dst_tensor.is_output == False:
                        inner_dst = inner_dst.setdefault(inner_dst.__len__()+1,name)
                    # tensor is output
                    elif dst_tensor.is_output is not None and dst_tensor.is_output == True:
                        for values in inner_dst.values():
                            if abs(int(values.split('_',1)[0]) - int(name.split('_',1)[0])) < 2000:
                                name = ''
                                break
                        if name != '':
                            inner_dst.setdefault(inner_dst.__len__()+1,name)

    @classmethod
    def process_functionalize_name(cls, name_split):
        """ process @_functionalize in string """
        if re.search("functionalize", name_split):
            if re.search(":", name_split):
                name_split = name_split.split("@_functionalize",1)[0] + ":" + name_split.split(":",1)[-1]
            else:
                name_split = name_split.split("@_functionalize",1)[0] + name_split.split("@",1)[1][-3:]
        return name_split

    @classmethod
    def process_inference_name(cls, name_split):
        """ process @__inference in string """
        if re.search("inference", name_split):
            if re.search(":", name_split):
                name_split = name_split.split("@__inference",1)[0] + ":" + name_split.split(":",1)[-1]
            else:
                name_split = name_split.split("@__inference",1)[0] + name_split.split("@",1)[1][-3:]
        return name_split

    @classmethod
    def process_stateful_name(cls, name_split):
        """ process StatefulPartitionedCall_StatefulPartitionedCall in string """
        stateful_str = "StatefulPartitionedCall_StatefulPartitionedCall_"
        if re.search(stateful_str, name_split):
            name_split = name_split.replace(stateful_str, "")
        return name_split

    @classmethod
    def process_at_name(cls, name_split):
        """ process at in string """
        if re.search("@", name_split):
            if re.search(":", name_split):
                name_split = name_split.split("@",1)[0] + ":" + name_split.split(":",1)[-1]
            else:
                name_split = name_split.split("@",1)[0] + name_split.split("@",1)[1][-3:]
        return name_split

    def process_string_name(self, name_split):
        self.preprocess_string(name_split)
        text_tail = re.compile(r".*_[0-9]$")
        if re.search("@axis", name_split):
            dst_file_name_split = str(name_split).split("@axis")
            if len(dst_file_name_split) > 2:
                raise ValueError(print_red_color("There are many @axis in pb name, we need 0 or 1 but get {}".format(name_split)))
            else:
                # Determine if it ends with "_0"
                if text_tail.match(dst_file_name_split[0]):
                    name_split = dst_file_name_split[0][::-1].replace("_",":",1)[::-1] + dst_file_name_split[1][-3:]
                else:
                    name_split = dst_file_name_split[0] + dst_file_name_split[1][-3:]
        # process some string we can not use
        name_split = self.process_functionalize_name(name_split)
        name_split = self.process_inference_name(name_split)
        name_split = self.process_stateful_name(name_split)
        name_split = self.process_at_name(name_split)
        return name_split

class PtProcess(AllFrameworkProcess):

    def process_src_data(self):
        num_record = {}
        for i ,src_file in enumerate(self.comp_data.src_files):
            if not src_file.endswith(".pb"):
                raise ValueError(print_red_color("Input file format error, only support .pb file."))
            src_tensor_name_split = self.src_preprocess(i)
            src_tensor_name_split = self.process_string_name(src_tensor_name_split)
            num_record[src_tensor_name_split] = num_record.get(src_tensor_name_split, 0) + 1
            self.comp_data.src_loop_cnt[i] = num_record[src_tensor_name_split]
            self.comp_data.src_record[src_file] = src_tensor_name_split

    def process_dst_data(self):
        for i, dst_file in enumerate(self.comp_data.dst_files):
            if not dst_file.endswith(".pb"):
                raise ValueError(print_red_color("Input file format error, only support .pb file."))
            dst_tensor_name_split = dst_file.split('_',1)[1]
            dst_tensor_name_split = self.process_string_name(dst_tensor_name_split)
            inner_dst = self.comp_data.dst_record.setdefault(dst_tensor_name_split, {})
            inner_dst.setdefault(inner_dst.__len__()+1, dst_file)

    def process_string_name(self, name_split):
        self.preprocess_string(name_split)
        dst_file_name_new = name_split
        dst_file_name_split = str(name_split).split("@axis")
        if len(dst_file_name_split) == 2:
            dst_file_name_new = dst_file_name_split[0] + dst_file_name_split[1][-3:]
        elif len(dst_file_name_split) > 2:
            raise ValueError(print_red_color("There are many @axis in pb name, we need 0 or 1 but get {}".format(name_split)))
        return dst_file_name_new

class OnnxProcess(AllFrameworkProcess):

    def process_src_data(self):
        num_record = {}
        for i ,src_file in enumerate(self.comp_data.src_files):
            if not src_file.endswith(".pb"):
                raise ValueError(print_red_color("Input file format error, only support .pb file."))
            src_tensor_name_split = self.src_preprocess(i)
            src_tensor_name_split = self.process_string_name(src_tensor_name_split)
            num_record[src_tensor_name_split] = num_record.get(src_tensor_name_split, 0) + 1
            self.comp_data.src_loop_cnt[i] = num_record[src_tensor_name_split]
            self.comp_data.src_record[src_file] = src_tensor_name_split

    def process_dst_data(self):
        for i, dst_file in enumerate(self.comp_data.dst_files):
            if not dst_file.endswith(".pb"):
                raise ValueError(print_red_color("Input file format error, only support .pb file."))
            dst_tensor_name_split = dst_file.split('_',1)[1]
            dst_tensor_name_split = self.process_string_name(dst_tensor_name_split)
            inner_dst = self.comp_data.dst_record.setdefault(dst_tensor_name_split, {})
            inner_dst.setdefault(inner_dst.__len__()+1, dst_file)

    def process_string_name(self, name_split):
        self.preprocess_string(name_split)
        dst_file_name_new = name_split
        dst_file_name_split = str(name_split).split("@axis")
        if len(dst_file_name_split) == 2:
            dst_file_name_new = dst_file_name_split[0] + dst_file_name_split[1][-3:]
        elif len(dst_file_name_split) > 2:
            raise ValueError(print_red_color("There are many @axis in pb name, we need 0 or 1 but get {}".format(name_split)))
        return dst_file_name_new

class CaffeProcess(AllFrameworkProcess):

    def process_src_data(self):
        num_record = {}
        for i ,src_file in enumerate(self.comp_data.src_files):
            if not src_file.endswith(".pb"):
                raise ValueError(print_red_color("Input file format error, only support .pb file."))
            src_tensor_name_split = self.src_preprocess(i)
            src_tensor_name_split = self.process_string_name(src_tensor_name_split)
            num_record[src_tensor_name_split] = num_record.get(src_tensor_name_split, 0) + 1
            self.comp_data.src_loop_cnt[i] = num_record[src_tensor_name_split]
            self.comp_data.src_record[src_file] = src_tensor_name_split

    def process_dst_data(self):
        for i, dst_file in enumerate(self.comp_data.dst_files):
            if not dst_file.endswith(".pb"):
                raise ValueError(print_red_color("Input file format error, only support .pb file."))
            dst_tensor_name_split = dst_file.split('_',1)[1]
            dst_tensor_name_split = self.process_string_name(dst_tensor_name_split)
            inner_dst = self.comp_data.dst_record.setdefault(dst_tensor_name_split, {})
            inner_dst.setdefault(inner_dst.__len__()+1, dst_file)

    def process_string_name(self, name_split):
        self.preprocess_string(name_split)
        dst_file_name_new = name_split
        dst_file_name_split = str(name_split).split("@axis")
        if len(dst_file_name_split) == 2:
            dst_file_name_new = dst_file_name_split[0] + dst_file_name_split[1][-3:]
        elif len(dst_file_name_split) > 2:
            raise ValueError(print_red_color("There are many @axis in pb name, we need 0 or 1 but get {}".format(name_split)))
        return dst_file_name_new
        
class DirectorFramework(object):

    def __init__(self, framework_process):
        self.framework_process = framework_process
    def construct(self):
        self.framework_process.dst_preprocess()
        self.framework_process.process_dst_data()
        self.framework_process.process_src_data()
        count_pass = self.framework_process.loop_compare()
        return count_pass

class process_data(object):
    @staticmethod
    def parse_trans_key(trans_key, rank):
        transpose_shape = []
        if trans_key != '':
            if re.match("(-?[0-9])to(-?[0-9])", trans_key):
                trans_split = trans_key.split("to")
                ori_dim = (int(trans_split[0]) + rank) % rank
                dst_dim = (int(trans_split[1]) + rank) % rank
                transpose_shape = [i for i in range(rank)]
                if ori_dim >= dst_dim:
                    transpose_shape.insert(ori_dim+1,dst_dim)
                    transpose_shape.pop(dst_dim)
                else:
                    transpose_shape.insert(ori_dim,dst_dim)
                    transpose_shape.pop(dst_dim+1)
                trans_split = trans_key.split("to")
            elif re.match("\[(\d)?(,\d)*\]", trans_key):
                trans_split = re.findall("\d+",trans_key)
                trans_split = list(map(int,trans_split))
                if len(set(trans_split)) != len(trans_split):
                    raise ValueError(print_red_color("The suffix of this tensor.pb is wrong, suffix = {}".format(trans_key)))
                for i in range(len(trans_split)):
                    if abs((int(trans_split[i]))) >=rank:
                        raise ValueError(print_red_color("The suffix of this tensor.pb is wrong, suffix = {}".format(trans_key)))
                    for j in range(len(trans_split)):
                        if trans_split[j]==i or trans_split[j]+len(trans_split)==i:
                            transpose_shape.append(j)
                            break
            else:
                raise ValueError(
                    print_red_color(
                        "The format of transkey is wrong, @axis[1,2,3,4] or @axis1to-1 is required, but get @axis{}"
                        .format(trans_key)))
        return tuple(transpose_shape)

    @staticmethod
    def cal_diff_formula(src_data, dst_data, zero_data, one_data, thre):
        """ calculate diff_formula """
        with np.errstate(divide='ignore', invalid='ignore'):
            if np.abs(np.subtract(src_data, dst_data)).sum() == 0:
                diff1 = 0.0
                diff2 = 0.0
                diff3_1 = 0.0
                diff3_2 = 0.0
            else:
                diff1 = np.abs(np.subtract(
                    src_data, dst_data)).sum() / (
                        np.abs(src_data).sum()
                        if src_data.sum() else 1e-9)
                diff2 = np.sqrt(
                    np.power(np.subtract(src_data, dst_data),
                             2).sum() / (np.power(src_data, 2).sum() if
                                         src_data.sum() else 1e-9))
                diff3_1 = np.where(np.abs(src_data) > thre,np.divide(np.abs(np.subtract(src_data,
                            dst_data)), np.abs(src_data)),
                            zero_data).max()
                diff3_2 = np.where(np.abs(src_data) <= thre,
                            np.abs(np.subtract(src_data, dst_data)),
                            zero_data).max()

        diff4_3 = one_data.sum() - (np.where(dst_data == src_data, one_data, zero_data).sum())
        diff4_1 = np.where(dst_data > src_data, one_data, zero_data).sum()
        diff4_2 = np.where(dst_data < src_data, one_data, zero_data).sum()
        if diff4_3 == 0:
            diff4_1 = 0.0
            diff4_2 = 0.0
        else:
            diff4_1 = diff4_1 / diff4_3
            diff4_2 = diff4_2 / diff4_3
        return diff1, diff2, diff3_1, diff3_2, diff4_1, diff4_2, diff4_3

    @staticmethod
    def cal_diff(dst_file_name, src_array, dst_array, src_trans_key, dst_trans_key):
        """ deal with origin data and return diffs """
        dst_transpose_shape = process_data.parse_trans_key(dst_trans_key, len(dst_array.shape))
        src_transpose_shape = process_data.parse_trans_key(src_trans_key, len(src_array.shape))
        if (dst_trans_key != '' and src_array.shape == dst_array.transpose(dst_transpose_shape).shape):
            print(print_green_color("dst tensor layout transpose: " + dst_file_name + ": " + dst_trans_key))
            dst_array = dst_array.transpose(dst_transpose_shape)
        elif (src_trans_key != '' and src_array.transpose(src_transpose_shape).shape == dst_array.shape):
            print(print_green_color("src tensor layout transpose: " + src_trans_key))
            src_array = src_array.transpose(src_transpose_shape)
        elif ((dst_trans_key != '' and src_trans_key != '')
              and src_array.transpose(src_transpose_shape).shape
              == dst_array.transpose(dst_transpose_shape).shape):
            print(print_green_color("dst tensor layout transpose: " + dst_file_name + ": " + dst_trans_key))
            print(print_green_color("src tensor layout transpose: " + src_trans_key))
            dst_array = dst_array.transpose(dst_transpose_shape)
            src_array = src_array.transpose(src_transpose_shape)
        if src_array.shape != dst_array.shape:
            print(print_yellow_color("###############\n dst and src tensor shape not equal: \n"+dst_file_name))
            return 999,999,[999,999],[999,999,999]
        if src_array.dtype == "float16" or dst_array.dtype == "float16":
            thre = 10**-4
        else:
            thre = 10**-6
        double_src_array = src_array.astype('float64')
        double_dst_array = dst_array.astype('float64')
        zero = np.zeros(src_array.shape).astype('float64')
        ones = np.ones(src_array.shape).astype('float64')
        diff1, diff2, diff3_1, diff3_2, diff4_1, diff4_2, diff4_3 = process_data.cal_diff_formula(
            double_src_array, double_dst_array, zero, ones, thre)

        return diff1, diff2, [diff3_1, diff3_2], [diff4_1, diff4_2, diff4_3]

    @staticmethod
    def get_array_from_tensor(src_tensor):
        src_list = []
        dtype = "float32"
        if src_tensor.dtype == types_pb2.DT_FLOAT32:
            src_val = src_tensor.float_val
            dtype = "float32"
        elif src_tensor.dtype == types_pb2.DT_FLOAT16:
            src_val = src_tensor.float_val
            dtype = "float16"
        elif src_tensor.dtype == types_pb2.DT_FLOAT64:
            src_val = src_tensor.double_val
            dtype = "float64"
        elif src_tensor.dtype == types_pb2.DT_INT32 or src_tensor.dtype == types_pb2.DT_INT16 \
                or src_tensor.dtype == types_pb2.DT_INT8 or src_tensor.dtype == types_pb2.DT_INT4 \
                or src_tensor.dtype == types_pb2.DT_BOOL:
            src_val = src_tensor.int_val
            dtype = "int32"
        elif src_tensor.dtype == types_pb2.DT_INT64:
            src_val = src_tensor.int64_val
            dtype = "int64"
        elif src_tensor.dtype == types_pb2.DT_UINT32 or src_tensor.dtype == types_pb2.DT_UINT16 \
                or src_tensor.dtype == types_pb2.DT_UINT8 or src_tensor.dtype == types_pb2.DT_UINT4:
            src_val = src_tensor.uint32_val
            dtype = "uint32"
        elif src_tensor.dtype == types_pb2.DT_UINT64:
            src_val = src_tensor.uint64_val
            dtype = "uint64"
        else:
            print(print_yellow_color("Dtype :" + str(src_tensor.dtype) + " is unsupported yet."))
            return np.ones(src_tensor.tensor_shape)
        for i in src_val:
            src_list.append(i)
        if len(src_list) != 0:
            src_array = np.array(src_list).reshape(src_tensor.tensor_shape).astype(dtype)
        else:
            src_array = np.empty(src_tensor.tensor_shape)
        return src_array

def compare_data(src_dir, dst_dir, output_dir, framework):
    if framework == "tensorflow":
        framework_process = TfProcess(src_dir, dst_dir, output_dir)
    elif framework == "pytorch":
        framework_process = PtProcess(src_dir, dst_dir, output_dir)
    elif framework == "caffe":
        framework_process = CaffeProcess(src_dir, dst_dir, output_dir)
    elif framework == "onnx":
        framework_process = OnnxProcess(src_dir, dst_dir, output_dir)
    else:
        raise ValueError(print_red_color("Please enter the correct framework name, now the framework name is {}".format(framework)))
    run_diff = DirectorFramework(framework_process)
    count_pass = run_diff.construct()
    return count_pass

class OperationHandle(object):
    def __init__(self, args):
        self.args = args
    def operation():
        raise NotImplementedError

class CompareHandle(OperationHandle):
    def __init__(self, args):
        super(CompareHandle, self).__init__(args)
        self.src_dir = self.args.src_dir
        self.dst_dir = self.args.dst_dir
        self.output_dir = self.args.output_dir
        self.framework = self.args.framework
    def operation(self):
        match_num = compare_data(self.src_dir, self.dst_dir, self.output_dir, self.framework)
        return match_num
class ConvertHandle(OperationHandle):
    def __init__(self, args):
        super(ConvertHandle, self).__init__(args)
        self.input_file = self.args.input_file
        self.output_file = self.args.output_file
    def operation(self):
        src_tensor = tensor_pb2.TensorProto()
        with open(self.input_file, 'rb') as in_file:
            src_tensor.ParseFromString(in_file.read())
        with open(self.output_file, 'w') as out_file:
            print(src_tensor, file=out_file)
        return

class SingleHandle(OperationHandle):
    def __init__(self, args):
        super(SingleHandle, self).__init__(args)
        self.src_file = self.args.src_file
        self.dst_file = self.args.dst_file
        self.mode = self.args.mode
        self.th1 = self.args.th1
        self.th2 = self.args.th2
        self.result_file = self.args.result_file
        self.trans_key = self.args.trans_key
    def operation(self):
        src_tensor = tensor_pb2.TensorProto()
        dst_tensor = tensor_pb2.TensorProto()
        with open(self.src_file, 'rb') as s_file:
            src_tensor.ParseFromString(s_file.read())
        with open(self.dst_file, 'rb') as d_file:
            dst_tensor.ParseFromString(d_file.read())

        src_npy = process_data.get_array_from_tensor(src_tensor)
        dst_npy = process_data.get_array_from_tensor(dst_tensor)
        transpose_shape = process_data.parse_trans_key(self.trans_key, len(src_npy.shape))
        if ( self.trans_key != '' and src_npy.shape == dst_npy.transpose(transpose_shape).shape):
            print("[Compare Single] dst tensor layout transpose: Layout "+self.trans_key[:self.trans_key.index("2")]+ "===>"+self.trans_key[self.trans_key.index("2")+1:])
            dst_npy = dst_npy.transpose(transpose_shape)

        if src_npy.shape != dst_npy.shape:
            print(print_red_color("Dst and src tensor shape are not equal! To Exit."))
            exit(0)
        diff1, diff2, diff3, diff4 = process_data.cal_diff("From compare single", src_npy, dst_npy, '', '')

        errs = []
        for i in range(src_npy.size):
            index = self.get_index(i, src_npy.shape)
            src_num = src_npy.ravel()[i]
            dst_num = dst_npy.ravel()[i]
            d1 = abs(dst_num - src_num)
            d2 = abs((dst_num - src_num) / src_num)
            if ((self.mode == 1 and d1 > self.th1) or
            (self.mode == 2 and d2 > self.th2) or
            (self.mode == 3 and (d1 > self.th1 or d2 > self.th2)) or
            (self.mode == 4 and (d1 > self.th1 and d2 > self.th2))) :
                errs.append([index, d1, d2])

        with open(self.result_file, 'w') as f:
            f.write("Shape: "+str(src_npy.shape)+'\n')
            f.write("Diff1: "+str(diff1)+'\n')
            f.write("Diff2: "+str(diff2)+'\n')
            f.write("Diff3: "+str(diff3)+'\n')
            f.write("Diff4: "+str(diff4)+'\n')
            f.writelines([str(err)+'\n' for err in errs])
        return True
    
    def get_index(self, num, shape):
        res = []
        shape = list(shape)
        shape.reverse()
        for i in shape:
            res.append(num % i)
            num = num // i
        res.reverse()
        return res

class OperationFactory(object):
    def process(self, args):
        if args.name == "compare":
            get_operation_results = CompareHandle(args)
        elif args.name == "convert":
            get_operation_results = ConvertHandle(args)
        elif args.name == "single":
            get_operation_results = SingleHandle(args)
        else:
            raise ValueError(print_red_color("Please input correct name, like compare/convert/single, now is {}".format(args.name)))
        get_operation_results.operation()


def create_parser():
    parser = argparse.ArgumentParser(description="Pb compare and convert tools.")

    subparsers = parser.add_subparsers(help='commands')

    compare_parser = subparsers.add_parser(name='compare', help='Compare two directories of pb files.')
    compare_parser.add_argument('-s', '--src_dir', required=True, help="Source framwork dump tensor data path.")
    compare_parser.add_argument('-d', '--dst_dir', required=True, help="Destination dump tensor data path.")
    compare_parser.add_argument('-o', '--output_dir', default="diff_jsons", help="Directory to dump diff json.")
    compare_parser.add_argument('--framework', required=True, default='tensorflow', help="Set compare source framework")
    compare_parser.set_defaults(name="compare")

    convert_parser = subparsers.add_parser(name='convert', help='Convert one pb file to readable file.')
    convert_parser.add_argument('-i', '--input_file', required=True, help="Source input pb file.")
    convert_parser.add_argument('-o', '--output_file', default="output_file", help="Path to save output file.")
    convert_parser.set_defaults(name="convert")

    single_parser = subparsers.add_parser(name='single', help='Convert one pb file to readable file.')
    single_parser.add_argument('-s', '--src_file', required=True, help="Source input pb file.")
    single_parser.add_argument('-d', '--dst_file', required=True, help="Destination input pb file.")
    single_parser.add_argument('--th1', default=1, type=float, help="Relative error threshold.")
    single_parser.add_argument('--th2', default=0.03, type=float, help="Absolute error threshold.")
    single_parser.add_argument('-m', '--mode', default=1, type=int, help="Output mode. Mode 1: gt th1; Mode 2: gt th2; Mode 3: gt th1 or th2; Mode 4: gt th1 and th2.")
    single_parser.add_argument('-o', '--result_file', default="./result_file", help="Path to save result file.")
    single_parser.add_argument('-k', '--trans_key', default="", help="HWCN2NHWC or NCHW2NHWC. use to transpose dst tensor.")
    single_parser.set_defaults(name="single")
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    operation_factory = OperationFactory()
    try:
        operation_factory.process(args)
    except AttributeError:
        parser.error("too few arguments, add -h to check usage.")
