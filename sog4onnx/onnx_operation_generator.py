#! /usr/bin/env python

import sys
import os
import json
import traceback
from argparse import ArgumentParser
import onnx
import onnx_graphsurgeon as gs
import numpy as np
from onnx_graphsurgeon.ir.tensor import Constant
from onnxsim import simplify
from typing import Dict, Optional, List

class Color:
    BLACK          = '\033[30m'
    RED            = '\033[31m'
    GREEN          = '\033[32m'
    YELLOW         = '\033[33m'
    BLUE           = '\033[34m'
    MAGENTA        = '\033[35m'
    CYAN           = '\033[36m'
    WHITE          = '\033[37m'
    COLOR_DEFAULT  = '\033[39m'
    BOLD           = '\033[1m'
    UNDERLINE      = '\033[4m'
    INVISIBLE      = '\033[08m'
    REVERCE        = '\033[07m'
    BG_BLACK       = '\033[40m'
    BG_RED         = '\033[41m'
    BG_GREEN       = '\033[42m'
    BG_YELLOW      = '\033[43m'
    BG_BLUE        = '\033[44m'
    BG_MAGENTA     = '\033[45m'
    BG_CYAN        = '\033[46m'
    BG_WHITE       = '\033[47m'
    BG_DEFAULT     = '\033[49m'
    RESET          = '\033[0m'


def generate():
    pass

def main():
    # https://github.com/onnx/onnx/blob/main/docs/Operators.md

    parser = ArgumentParser()
    parser.add_argument(
        '--op_type',
        type=str,
        required=True,
        default='',
        help='ONNX OP type. https://github.com/onnx/onnx/blob/main/docs/Operators.md'
    )
    parser.add_argument(
        '--opset',
        type=int,
        required=True,
        help='ONNX opset number.'
    )
    """
    python3 onnx_operation_generator.py \
    --op_type Gemm \
    --opset 11 \
    --input_variables i1 np.float32 [1,2,3] \
    --input_variables i2 np.float32 [1,1] \
    --input_variables i3 np.float32 0 \
    --attributes "{\"alpha\": 1.0, \"beta\": 1.0, \"transA\": 0, \"transB\": 0}"
    """
    parser.add_argument(
        '--input_variables',
        type=str,
        required=True,
        nargs='+',
        action='append',
        help=\
            'input_variables can be specified multiple times. \n'+
            '--input_variables variable_name numpy.dtype shape \n\n'+
            'e.g.\n'+
            '--input_variables i1 float32 [1,3,5,5] \n'+
            '--input_variables i2 int32 [1] \n'+
            '--input_variables i3 float64 [1,3,224,224]'
    )
    parser.add_argument(
        '--attributes',
        type=json.loads,
        help=\
            'attributes can be specified multiple times. \n'+
            'The key name is a string and the delimiter is double-cotation marks. \n'+
            'Note that double-cotation marks must be escaped with a backslash. \n'+
            '--attributes {"attribute_name1": value1, "attribute_name2": value2, ...} \n\n'+
            'e.g.\n'+
            '--attributes "{\\"alpha\\": 1.0, \\"beta\\": 1.0, \\"transA\\": 0, \\"transB\\": 0}"'
    )
    parser.add_argument(
        '--output_variables',
        type=str,
        required=True,
        nargs='+',
        action='append',
        help=\
            'output_variables can be specified multiple times. \n'+
            '--output_variables variable_name numpy.dtype shape \n\n'+
            'e.g.\n'+
            '--output_variables o1 float32 [1,3,5,5] \n'+
            '--output_variables o2 int32 [1] \n'+
            '--output_variables o3 float64 [1,3,224,224]'
    )
    parser.add_argument(
        '--output_onnx_file_path',
        type=str,
        default='',
        help=\
            'Output onnx file path. If not specified, a file with the OP type name is generated.'+
            'e.g. op_type="Gemm" -> Gemm.onnx'
    )
    args = parser.parse_args()
    from pprint import pprint

    # 1. Parameter parsing
    ## input variables
    """
    input_variables = {'name': [dtype, shape]}
    """
    input_variables = {input_variable[0]: [getattr(np, input_variable[1]), eval(input_variable[2])] for input_variable in args.input_variables}
    """
    input_gs_variables
    [
        Variable (i1): (shape=[1, 2, 3], dtype=<class 'numpy.float32'>),
        Variable (i2): (shape=[1, 1], dtype=<class 'numpy.float32'>),
        Variable (i3): (shape=[0], dtype=<class 'numpy.int32'>)
    ]
    """
    input_gs_variables = [gs.Variable(name=key, dtype=value[0], shape=value[1]) for key, value in input_variables.items()]

    ## output variables
    """
    output_variables = {'name': [dtype, shape]}
    """
    output_variables = {output_variable[0]: [getattr(np, output_variable[1]), eval(output_variable[2])] for output_variable in args.output_variables}
    """
    output_gs_variables
    [
        Variable (i1): (shape=[1, 2, 3], dtype=<class 'numpy.float32'>),
        Variable (i2): (shape=[1, 1], dtype=<class 'numpy.float32'>),
        Variable (i3): (shape=[0], dtype=<class 'numpy.int32'>)
    ]
    """
    output_gs_variables = [gs.Variable(name=key, dtype=value[0], shape=value[1]) for key, value in output_variables.items()]

    # 2. Node Generation
    node = gs.Node(
        op=args.op_type,
        attrs=args.attributes,
        inputs=input_gs_variables,
        outputs=output_gs_variables
    )

    # 3. Graph Generation
    graph = gs.Graph(
        nodes=[node],
        inputs=input_gs_variables,
        outputs=output_gs_variables,
        opset=args.opset,
    )
    model_def = gs.export_onnx(graph)

    # 4. Graph Check
    try:
        onnx.checker.check_model(
            model=model_def,
            full_check=False
        )
        print(f'{Color.GREEN}INFO:{Color.RESET} The model is checked!')
    except Exception as e:
        tracetxt = traceback.format_exc().splitlines()[-1]
        print(f'{Color.RED}ERROR:{Color.RESET} {tracetxt}')

    # 5. Save
    if args.output_onnx_file_path:
        onnx.save(gs.export_onnx(graph), args.output_onnx_file_path)
    else:
        onnx.save(gs.export_onnx(graph), f"{args.op_type}.onnx")

if __name__ == '__main__':
    main()