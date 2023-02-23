#!/usr/bin/env python

import argparse
import numpy as np
import os
import sys, ast
import torch
try:
  from magicmind.tools.debug_tools.pytorch_tools import set_dump_dir, insert_placeholder, insert_sim_quant, insert_sim_quant_fp16, force_fp16, fusebn
  from magicmind.tools.debug_tools.pytorch_tools.fake_quantize import *
  from magicmind.tools.debug_tools.utils.common_utils import *

except:
  sys.path.append(os.path.join(os.path.dirname(__file__), "lib/pytorch_tools"))
  import set_dump_dir, insert_placeholder, insert_sim_quant
  from fake_quantize import *

logger = get_logger()

class PytorchDump:
    def __init__(self, args, qparams_json = "", use_mm_calib = False):
        self.traced = None
        self.processed_model = ""
        self.out = []
        self.__args = args
        self.__qparams_json = qparams_json
        self.__use_mm_calib = use_mm_calib
        self.__pytorch_dump_dir = ""
        self.__input_tensors = []
        self.__real_input_shapes = []
        self.__dtypes = []
        self.__valid_quant_precision_mode = ["qint8_mixed_float32", "qint16_mixed_float32",
                                             "qint8_mixed_float16", "qint16_mixed_float16"]
        self.__valid_precision_mode = ["force_float32", "force_float16"] + self.__valid_quant_precision_mode

    def __check_parameters(self):
        if self.__args.precision_mode not in self.__valid_precision_mode:
            raise ValueError(print_red_color("{} is illegal! Support precision mode: '{}'," \
                    " '{}', '{}', '{}', '{}', '{}'.".format(self.__args.precision_mode, *self.__valid_precision_mode)))

        if self.__args.dump_dir == "" or self.__args.dump_model_dir == "":
            raise ValueError(print_red_color("ERROR: dump_dir or dump_model_dir is empty."))

        # TODO(liuxuewen): Wait for the external interface to be perfected before providing it to the user 
        if (self.__args.precision_mode not in ["force_float32", "force_float16"]) \
            and (self.__args.calibrate_inputs_dirs == "") \
            and (self.__qparams_json == ""):
            raise ValueError(print_red_color("Fake quant mode and calibrate inputs are invalid."))

        if self.__qparams_json != "" and self.__args.calibrate_inputs_dirs != "":
            raise ValueError(print_red_color("Please make sure there is only one input for quantize_params_json and calibrate_inputs_dirs."))

        if (self.__args.precision_mode in self.__valid_quant_precision_mode) \
            and (self.__args.calibrate_inputs_dirs != ""):
            self.__use_mm_calib = True

        if self.__args.input_channel_quantize == True and "DwConv" not in self.__args.int_ops:
            print(print_yellow_color("The parameter input_channel_quantize takes effect " \
                                    + "only when the DwConv operator in int_ops"))
            self.__args.input_channel_quantize = False

        if self.__args.input_channel_quantize == True and self.__args.asymmetric_quantize == True:
            print(print_yellow_color("tools doesn't support both input_chahnel and input_asymmetric," \
                                    + "convert input_asymmetric = True to input_asymmetric = False by default"))
            self.__args.asymmetric_quantize = False

    def __process_dump_dir(self):
        if not os.path.exists(self.__args.dump_dir):
            print("make dir: {}".format(self.__args.dump_dir))
            os.makedirs(self.__args.dump_dir)

        if not os.path.exists(self.__args.dump_model_dir):
            print("make dir: {}".format(self.__args.dump_model_dir))
            os.makedirs(self.__args.dump_model_dir)

        if (self.__args.tensor_format == "pbtxt"):
            os.environ['SAVE_AS_PBTXT'] = "1"
        
        self.__pytorch_dump_dir = self.__args.dump_dir + '/pytorch_dump_dir'
        os.system("mkdir -vp " + self.__args.dump_dir)
        if os.path.isdir(self.__pytorch_dump_dir):
            os.system("rm -r " + self.__pytorch_dump_dir)
        os.system("mkdir " + self.__pytorch_dump_dir)
        
        self.processed_model = os.path.join(self.__args.dump_model_dir, 'pytorch_processed_model.pt')

    def __check_and_process_parameters(self):
        self.__check_parameters()
        self.__process_dump_dir()
        return self
    
    def __process_input_shapes(self, number_of_inputs):
        input_shapes_list = []
        if self.__args.input_shapes != ["None"]:
            if not isinstance(self.__args.input_shapes, list):
                self.__args.input_shapes = ast.literal_eval(self.__args.input_shapes)
            for i in self.__args.input_shapes:
                if not isinstance(i, list):
                    input_shapes_list.append(ast.literal_eval(i))
                else:
                    input_shapes_list.append(i)
            check_input_shape_and_files(input_shapes_list, self.__args.bin_files, number_of_inputs)
        else:
            check_input_shape_and_files(self.__args.input_shapes, self.__args.bin_files, number_of_inputs)
        return input_shapes_list
    
    def __convert_input_to_tensor(self):
        input_len = len(list(self.traced.graph.inputs())) - 1
        input_shapes_list = self.__process_input_shapes(input_len)
        data_types = []
        for i, input_file in enumerate(self.__args.bin_files):
            input_array = np.load(input_file)
            data_types.append(str(input_array.dtype))
            self.__real_input_shapes.append(list(input_array.shape))
            if self.__args.input_shapes != ["None"]:
                check_shape_equal(input_shapes_list[i], list(input_array.shape))
            input_tensor = torch.from_numpy(input_array)
            self.__input_tensors.append(input_tensor)
        self.__dtypes = process_dtype(data_types, input_len)
        return self
    
    def __load_model(self):
        self.traced = torch.jit.load(self.__args.model_file)
        set_dump_dir(self.__pytorch_dump_dir)
        self.__convert_input_to_tensor()
        if self.__args.fold_conv_bn:
            self.traced = fusebn(self.traced._c)
            torch.jit.save(self.traced, "sim_quant2.pt")
            self.traced = torch.jit.load("sim_quant2.pt")
        insert_placeholder(self.traced.graph, self.__args.dump_all)
        os.system("mv " + self.__pytorch_dump_dir + "/input_graph.txt " + self.__args.dump_model_dir)
        torch.jit.save(self.traced, self.processed_model)
        return self

    def __fake_quant_model(self):
        if self.__use_mm_calib:
            network = create_calib_network(self.processed_model, "ModelKind.kPytorch", 
                                           self.__args.calibrate_inputs_dirs, 
                                           self.__real_input_shapes,
                                           self.__args.quantize_algorithm,
                                           self.__args.int_ops,
                                           self.__args.input_channel_quantize,
                                           self.__args.filter_channel_quantize,
                                           self.__args.asymmetric_quantize,
                                           True, True, 
                                           self.__args.remote_address, 
                                           self.__dtypes)
            sim_quant_model = fake_quant(self.traced, network,
                                         QUANTBIT_MAP[self.__args.precision_mode],
                                         self.__args.precision_mode,
                                         self.__args.int_ops,
                                         self.__args.input_channel_quantize,
                                         self.__args.filter_channel_quantize,
                                         self.__args.asymmetric_quantize)
        else:
            json_content = parse_json(self.__qparams_json)
            sim_quant_model = get_data_from_json(self.traced, json_content, 
                                                 QUANTBIT_MAP[self.__args.precision_mode],
                                                 self.__args.precision_mode,
                                                 self.__args.int_ops,
                                                 self.__args.input_channel_quantize,
                                                 self.__args.filter_channel_quantize,
                                                 self.__args.asymmetric_quantize)
        return  sim_quant_model
    
    def __run_model(self):
        if self.__args.precision_mode == "force_float32":
            self.out = self.traced(*(self.__input_tensors))
        elif self.__args.precision_mode == "force_float16":
            force_fp16(self.traced.graph)
            self.out = self.traced(*(self.__input_tensors))
        elif self.__args.precision_mode in self.__valid_quant_precision_mode:
            sim_quant_model = self.__fake_quant_model()
            self.out = sim_quant_model(*(self.__input_tensors))
        logger.info("=============================tensor dump finished=============================")
        return self
            
    def fake_quant(self):
        self.__check_and_process_parameters().__load_model().__run_model()
        return self.out
    
    def dump(self):
        self.__check_and_process_parameters() \
            .__load_model() \
            .__run_model()

def create_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_file", type=str, required=True, help="config path for pt model file, e.g /tmp/model.pt")
    parser.add_argument(
        "--bin_files", type=str, nargs= '+', required=True, help="config path hold image binary files")
    parser.add_argument(
        "--dump_dir", type=str, default="./", help="config dump directory to save result, e.g /tmp/pytorch_dump_dir/")
    parser.add_argument(
        "--dump_model_dir", type=str, default="./", help="config dump directory to save the preprocessed pt model file, e.g /tmp")
    parser.add_argument(
        "--input_shapes", type=str, nargs='+', default="None", help="shapes of input tensor")
    parser.add_argument("--tensor_format", type=str, default="pb", help="tensor save format: 'pb' or 'pbtxt' ") 
    parser.add_argument(
        "--dump_all", type=str2bool, default=True, help="only dump output or dump all nodes: 'True' or 'False' ")
    parser.add_argument("--precision_mode", type=str, default="force_float32", help="precision mode: 'force_float32', 'force_float16', \
        'qint8_mixed_float32', 'qint16_mixed_float32', 'qint8_mixed_float16', 'qint16_mixed_float16' ")
    parser.add_argument(
        "--int_ops", type=str, nargs='+', default=["Conv", "FC", "DeConv"], \
	help="which op can support fake_quantize")
    parser.add_argument(
        "--input_channel_quantize", type=str2bool, default=False, help="input channel quantization")
    parser.add_argument(
        "--filter_channel_quantize", type=str2bool, default=False, help="filter channel quantization")
    parser.add_argument("--asymmetric_quantize", type=str2bool, default=False, help="asymmetric quantization")
    parser.add_argument("--quantize_algorithm", type=str, default="linear", help="expected quantize algorithm to get quant_param")
    parser.add_argument("--calibrate_inputs_dirs", nargs='+', type=str, default="", help="directory of binary files for calibrate") 
    parser.add_argument("--remote_address", type=str, default=None,
            help = "The address of rpc server where the remote calibrator will be connect and running on, which should be in format 'ip: port' ")
    parser.add_argument("--fold_conv_bn", type=str2bool, default=False,
            help = "Switch for PyTorch conv BN fusion: 'True' or 'False' ")
    
    return parser

def main():
    parser = create_args_parser()
    args = parser.parse_args()
    pytorch_dump = PytorchDump(args)
    pytorch_dump.dump()

if __name__ == '__main__':
    main()