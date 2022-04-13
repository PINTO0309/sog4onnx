#! /usr/bin/env python

import json
import traceback
from argparse import ArgumentParser
import onnx
import onnx_graphsurgeon as gs
import numpy as np
from typing import Optional

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


def generate(
    op_type: str,
    opset: int,
    input_variables: Optional[dict] = None,
    output_variables: Optional[dict] = None,
    attributes: Optional[dict] = None,
    output_onnx_file_path: Optional[str] = '',
    non_verbose: Optional[bool] = False,
) -> onnx.ModelProto:
    """
    Parameters
    ----------
    op_type: str
        ONNX op type.\n\
        See below for the types of OPs that can be specified.\n\n\
        e.g. "Add", "Div", "Gemm", ...\n\
        https://github.com/onnx/onnx/blob/main/docs/Operators.md

    opset: int
        ONNX opset number.\n\
        e.g. 11

    input_variables: Optional[dict]
        Specify input variables for the OP to be generated.\n\
        See below for the variables that can be specified.\n\n\
        {"input_var_name1": [numpy.dtype, shape], "input_var_name2": [dtype, shape], ...}\n\n\
        e.g. input_variables = {"name1": [np.float32, [1,224,224,3]], "name2": [np.bool_, [0]], ...}\n\
        https://github.com/onnx/onnx/blob/main/docs/Operators.md

    output_variables: Optional[dict]
        Specify output variables for the OP to be generated.\n\
        See below for the variables that can be specified.\n\n\
        {"output_var_name1": [numpy.dtype, shape], "output_var_name2": [dtype, shape], ...}\n\n\
        e.g. output_variables = {"name1": [np.float32, [1,224,224,3]], "name2": [np.bool_, [0]], ...}\n\
        https://github.com/onnx/onnx/blob/main/docs/Operators.md

    attributes: Optional[dict]
        Specify output attributes for the OP to be generated.\n\
        See below for the attributes that can be specified.\n\n\
        {"attr_name1": value1, "attr_name2": value2, "attr_name3": value3, ...}\n\n\
        e.g. attributes = {"alpha": 1.0, "beta": 1.0, "transA": 0, "transB": 0}\n\
        Default: None\n\
        https://github.com/onnx/onnx/blob/main/docs/Operators.md

    output_onnx_file_path: Optional[str]
        Output of onnx file path.\n\
        If not specified, no .onnx file is output.\n\
        Default: ''

    non_verbose: Optional[bool]
        Do not show all information logs. Only error logs are displayed.\n\
        Default: False

    Returns
    -------
    single_op_graph: onnx.ModelProto
        Single op onnx ModelProto
    """

    # 1. Parameter parsing
    """
    input_gs_variables
    [
        Variable (i1): (shape=[1, 2, 3], dtype=<class 'numpy.float32'>),
        Variable (i2): (shape=[1, 1], dtype=<class 'numpy.float32'>),
        Variable (i3): (shape=[0], dtype=<class 'numpy.int32'>)
    ]
    """
    input_gs_variables = None
    if input_variables:
        input_gs_variables = [gs.Variable(name=key, dtype=value[0], shape=value[1]) for key, value in input_variables.items()]

    """
    output_gs_variables
    [
        Variable (i1): (shape=[1, 2, 3], dtype=<class 'numpy.float32'>),
        Variable (i2): (shape=[1, 1], dtype=<class 'numpy.float32'>),
        Variable (i3): (shape=[0], dtype=<class 'numpy.int32'>)
    ]
    """
    output_gs_variables = None
    if output_variables:
        output_gs_variables = [gs.Variable(name=key, dtype=value[0], shape=value[1]) for key, value in output_variables.items()]

    # 2. Node Generation
    node = gs.Node(
        op=op_type,
        attrs=attributes,
        inputs=input_gs_variables,
        outputs=output_gs_variables
    )

    # 3. Graph Generation
    graph = gs.Graph(
        nodes=[node],
        inputs=input_gs_variables,
        outputs=output_gs_variables,
        opset=opset,
    )
    single_op_graph = gs.export_onnx(graph)

    # 4. Graph Check
    try:
        onnx.checker.check_model(
            model=single_op_graph,
            full_check=False
        )
        if not non_verbose:
            print(f'{Color.GREEN}INFO:{Color.RESET} The model is checked!')

    except Exception as e:
        tracetxt = traceback.format_exc().splitlines()[-1]
        print(f'{Color.RED}ERROR:{Color.RESET} {tracetxt}')

    # 5. Save
    if output_onnx_file_path:
        onnx.save(single_op_graph, output_onnx_file_path)

    # 6. Return
    return single_op_graph


def main():
    # https://github.com/onnx/onnx/blob/main/docs/Operators.md
    parser = ArgumentParser()
    parser.add_argument(
        '--op_type',
        type=str,
        required=True,
        default='',
        help=\
            'ONNX OP type. \n'+
            'https://github.com/onnx/onnx/blob/main/docs/Operators.md'
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
        nargs=3,
        action='append',
        help=\
            'input_variables can be specified multiple times. \n'+
            '--input_variables variable_name numpy.dtype shape \n'+
            'https://github.com/onnx/onnx/blob/main/docs/Operators.md \n\n'+
            'e.g.\n'+
            '--input_variables i1 float32 [1,3,5,5] \n'+
            '--input_variables i2 int32 [1] \n'+
            '--input_variables i3 float64 [1,3,224,224]'
    )
    parser.add_argument(
        '--output_variables',
        type=str,
        nargs=3,
        action='append',
        help=\
            'output_variables can be specified multiple times. \n'+
            '--output_variables variable_name numpy.dtype shape \n'+
            'https://github.com/onnx/onnx/blob/main/docs/Operators.md \n\n'+
            'e.g.\n'+
            '--output_variables o1 float32 [1,3,5,5] \n'+
            '--output_variables o2 int32 [1] \n'+
            '--output_variables o3 float64 [1,3,224,224]'
    )
    parser.add_argument(
        '--attributes',
        nargs=2,
        action='append',
        help=\
            'attributes can be specified multiple times. \n'+
            '--attributes name value \n'+
            'https://github.com/onnx/onnx/blob/main/docs/Operators.md \n\n'+
            'e.g.\n'+
            '--attributes alpha 1.0 \n'+
            '--attributes beta 1.0 \n'+
            '--attributes transA 0 \n'+
            '--attributes transB 0'
    )
    parser.add_argument(
        '--output_onnx_file_path',
        type=str,
        default='',
        help=\
            'Output onnx file path. If not specified, a file with the OP type name is generated.'+
            'e.g. op_type="Gemm" -> Gemm.onnx'
    )
    parser.add_argument(
        '--non_verbose',
        action='store_true',
        help='Do not show all information logs. Only error logs are displayed.'
    )
    args = parser.parse_args()

    # input variables
    """
    input_variables_tmp = {'name': [dtype, shape]}
    """
    input_variables_tmp = None
    if args.input_variables:
        input_variables_tmp = {input_variable[0]: [getattr(np, input_variable[1]), eval(input_variable[2])] for input_variable in args.input_variables}

    # output variables
    """
    output_variables_tmp = {'name': [dtype, shape]}
    """
    output_variables_tmp = None
    if args.output_variables:
        output_variables_tmp = {output_variable[0]: [getattr(np, output_variable[1]), eval(output_variable[2])] for output_variable in args.output_variables}

    # attributes
    """
    attributes_tmp = {'name': value}
    """
    attributes_tmp = None
    if args.attributes:
        attributes_tmp = {attribute[0]: eval(attribute[1]) for attribute in args.attributes}

    # output_onnx_file_path
    output_onnx_file_path = ''
    if args.output_onnx_file_path:
        output_onnx_file_path = args.output_onnx_file_path
    else:
        output_onnx_file_path = f'{args.op_type}.onnx'

    # Generate
    single_op_graph = generate(
        op_type=args.op_type,
        opset=args.opset,
        input_variables=input_variables_tmp,
        output_variables=output_variables_tmp,
        attributes=attributes_tmp,
        output_onnx_file_path=output_onnx_file_path,
        non_verbose=args.non_verbose,
    )


if __name__ == '__main__':
    main()