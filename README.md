# sog4onnx
Simple ONNX operation generator. **S**imple **O**peration **G**enerator for **ONNX**.

https://github.com/PINTO0309/simple-onnx-processing-tools

[![Downloads](https://static.pepy.tech/personalized-badge/sog4onnx?period=total&units=none&left_color=grey&right_color=brightgreen&left_text=Downloads)](https://pepy.tech/project/sog4onnx) ![GitHub](https://img.shields.io/github/license/PINTO0309/sog4onnx?color=2BAF2B) [![PyPI](https://img.shields.io/pypi/v/sog4onnx?color=2BAF2B)](https://pypi.org/project/sog4onnx/) [![CodeQL](https://github.com/PINTO0309/sog4onnx/workflows/CodeQL/badge.svg)](https://github.com/PINTO0309/sog4onnx/actions?query=workflow%3ACodeQL)

<p align="center">
  <img src="https://user-images.githubusercontent.com/33194443/170155206-3e771286-b5c4-4ac0-a5d7-ec9a0196cbbb.png" />
</p>

# Key concept

- [x] Variable, Constant, Operation and Attribute can be generated externally.
- [x] Allow Opset to be specified externally.
- [x] No check for consistency of Operations within the tool, as new OPs are added frequently and the definitions of existing OPs change with each new version of ONNX's Opset.
- [x] Only one OP can be defined at a time, and the goal is to generate free ONNX graphs using a combination of **[snc4onnx](https://github.com/PINTO0309/snc4onnx)**, **[sne4onnx](https://github.com/PINTO0309/sne4onnx)**, **[snd4onnx](https://github.com/PINTO0309/snd4onnx)** and **[scs4onnx](https://github.com/PINTO0309/scs4onnx)**.
- [x] List of parameters that can be specified: https://github.com/onnx/onnx/blob/main/docs/Operators.md

## 1. Setup
### 1-1. HostPC
```bash
### option
$ echo export PATH="~/.local/bin:$PATH" >> ~/.bashrc \
&& source ~/.bashrc

### run
$ pip install -U onnx \
&& python3 -m pip install -U onnx_graphsurgeon --index-url https://pypi.ngc.nvidia.com \
&& pip install -U sog4onnx
```
### 1-2. Docker
https://github.com/PINTO0309/simple-onnx-processing-tools#docker

## 2. CLI Usage
```
$ sog4onnx -h

usage: sog4onnx [-h]
  --ot OP_TYPE
  --os OPSET
  --on OP_NAME
  [-iv NAME TYPE VALUE]
  [-ov NAME TYPE VALUE]
  [-a NAME DTYPE VALUE]
  [-of OUTPUT_ONNX_FILE_PATH]
  [-n]

optional arguments:
  -h, --help
    show this help message and exit

  -ot OP_TYPE, --op_type OP_TYPE
    ONNX OP type.
    https://github.com/onnx/onnx/blob/main/docs/Operators.md

  -os OPSET, --opset OPSET
    ONNX opset number.

  -on OP_NAME, --op_name OP_NAME
    OP name.

  -iv INPUT_VARIABLES INPUT_VARIABLES INPUT_VARIABLES, --input_variables INPUT_VARIABLES INPUT_VARIABLES INPUT_VARIABLES
    input_variables can be specified multiple times.
    --input_variables variable_name numpy.dtype shape
    https://github.com/onnx/onnx/blob/main/docs/Operators.md

    e.g.
    --input_variables i1 float32 [1,3,5,5] \
    --input_variables i2 int32 [1] \
    --input_variables i3 float64 [1,3,224,224]

  -ov OUTPUT_VARIABLES OUTPUT_VARIABLES OUTPUT_VARIABLES, --output_variables OUTPUT_VARIABLES OUTPUT_VARIABLES OUTPUT_VARIABLES
    output_variables can be specified multiple times.
    --output_variables variable_name numpy.dtype shape
    https://github.com/onnx/onnx/blob/main/docs/Operators.md

    e.g.
    --output_variables o1 float32 [1,3,5,5] \
    --output_variables o2 int32 [1] \
    --output_variables o3 float64 [1,3,224,224]

  -a ATTRIBUTES ATTRIBUTES ATTRIBUTES, --attributes ATTRIBUTES ATTRIBUTES ATTRIBUTES
    attributes can be specified multiple times.
    dtype is one of "float32" or "float64" or "int32" or "int64" or "str".
    --attributes name dtype value
    https://github.com/onnx/onnx/blob/main/docs/Operators.md

    e.g.
    --attributes alpha float32 1.0 \
    --attributes beta float32 1.0 \
    --attributes transA int32 0 \
    --attributes transB int32 0

  -of OUTPUT_ONNX_FILE_PATH, --output_onnx_file_path OUTPUT_ONNX_FILE_PATH
    Output onnx file path.
    If not specified, a file with the OP type name is generated.

    e.g. op_type="Gemm" -> Gemm.onnx

  -n, --non_verbose
    Do not show all information logs. Only error logs are displayed.
```

## 3. In-script Usage
```python
$ python
>>> from sog4onnx import generate
>>> help(generate)
Help on function generate in module sog4onnx.onnx_operation_generator:

generate(
  op_type: str,
  opset: int,
  op_name: str,
  input_variables: dict,
  output_variables: dict,
  attributes: Union[dict, NoneType] = None,
  output_onnx_file_path: Union[str, NoneType] = '',
  non_verbose: Union[bool, NoneType] = False
) -> onnx.onnx_ml_pb2.ModelProto

    Parameters
    ----------
    op_type: str
        ONNX op type.
        See below for the types of OPs that can be specified.
        https://github.com/onnx/onnx/blob/main/docs/Operators.md

        e.g. "Add", "Div", "Gemm", ...

    opset: int
        ONNX opset number.

        e.g. 11

    op_name: str
        OP name.

    input_variables: Optional[dict]
        Specify input variables for the OP to be generated.
        See below for the variables that can be specified.
        https://github.com/onnx/onnx/blob/main/docs/Operators.md
        {"input_var_name1": [numpy.dtype, shape], "input_var_name2": [dtype, shape], ...}

        e.g.
        input_variables = {
          "name1": [np.float32, [1,224,224,3]],
          "name2": [np.bool_, [0]],
          ...
        }

    output_variables: Optional[dict]
        Specify output variables for the OP to be generated.
        See below for the variables that can be specified.
        https://github.com/onnx/onnx/blob/main/docs/Operators.md
        {"output_var_name1": [numpy.dtype, shape], "output_var_name2": [dtype, shape], ...}

        e.g.
        output_variables = {
          "name1": [np.float32, [1,224,224,3]],
          "name2": [np.bool_, [0]],
          ...
        }

    attributes: Optional[dict]
        Specify output attributes for the OP to be generated.
        See below for the attributes that can be specified.
        When specifying Tensor format values, specify an array converted to np.ndarray.
        https://github.com/onnx/onnx/blob/main/docs/Operators.md
        {"attr_name1": value1, "attr_name2": value2, "attr_name3": value3, ...}

        e.g.
        attributes = {
          "alpha": 1.0,
          "beta": 1.0,
          "transA": 0,
          "transB": 0
        }
        Default: None

    output_onnx_file_path: Optional[str]
        Output of onnx file path.
        If not specified, no .onnx file is output.
        Default: ''

    non_verbose: Optional[bool]
        Do not show all information logs. Only error logs are displayed.
        Default: False

    Returns
    -------
    single_op_graph: onnx.ModelProto
        Single op onnx ModelProto
```

## 4. CLI Execution
```bash
$ sog4onnx \
--op_type Gemm \
--opset 1 \
--op_name gemm_custom1 \
--input_variables i1 float32 [1,2,3] \
--input_variables i2 float32 [1,1] \
--input_variables i3 int32 [0] \
--output_variables o1 float32 [1,2,3] \
--attributes alpha float32 1.0 \
--attributes beta float32 1.0 \
--attributes transA int32 0 \
--attributes transB int32 0
```

## 5. In-script Execution
```python
import numpy as np
from sog4onnx import generate

single_op_graph = generate(
    op_type = 'Gemm',
    opset = 1,
    op_name = "gemm_custom1",
    input_variables = {
      "i1": [np.float32, [1,2,3]],
      "i2": [np.float32, [1,1]],
      "i3": [np.int32, [0]],
    },
    output_variables = {
      "o1": [np.float32, [1,2,3]],
    },
    attributes = {
      "alpha": 1.0,
      "beta": 1.0,
      "broadcast": 0,
      "transA": 0,
      "transB": 0,
    },
    non_verbose = True,
)
```

## 6. Sample
### 6-1. opset=1, Gemm
```bash
$ sog4onnx \
--op_type Gemm \
--opset 1 \
--op_name gemm_custom1 \
--input_variables i1 float32 [1,2,3] \
--input_variables i2 float32 [1,1] \
--input_variables i3 int32 [0] \
--output_variables o1 float32 [1,2,3] \
--attributes alpha float32 1.0 \
--attributes beta float32 1.0 \
--attributes transA int32 0 \
--attributes transB int32 0
--non_verbose
```
![image](https://user-images.githubusercontent.com/33194443/163018526-f2d5c647-c3e9-4e65-9b9a-c1c4fa5da8a5.png)
![image](https://user-images.githubusercontent.com/33194443/163018647-a6880370-8772-4af1-9ffe-59820a621c30.png)

### 6-2. opset=11, Add
```bash
$ sog4onnx \
--op_type Add \
--opset 11 \
--op_name add_custom1 \
--input_variables i1 float32 [1,2,3] \
--input_variables i2 float32 [1,2,3] \
--output_variables o1 float32 [1,2,3] \
--non_verbose
```
![image](https://user-images.githubusercontent.com/33194443/163042479-9998ba73-ee26-44ea-bd6b-dcd04539190b.png)
![image](https://user-images.githubusercontent.com/33194443/163042529-5dbd1b5f-e8d1-47d0-8a9e-aacd91539c2b.png)

### 6-3. opset=11, NonMaxSuppression
```bash
$ sog4onnx \
--op_type NonMaxSuppression \
--opset 11 \
--op_name nms_custom1 \
--input_variables boxes float32 [1,6,4] \
--input_variables scores float32 [1,1,6] \
--input_variables max_output_boxes_per_class int64 [1] \
--input_variables iou_threshold float32 [1] \
--input_variables score_threshold float32 [1] \
--output_variables selected_indices int64 [3,3] \
--attributes center_point_box int64 1
```
![image](https://user-images.githubusercontent.com/33194443/163291737-8bd7ad7e-f9e5-4ce9-a8ba-444f1a8e77bb.png)
![image](https://user-images.githubusercontent.com/33194443/163291789-59e4e5c8-26f4-4971-ab22-1486093f1be0.png)

### 6-4. opset=11, Constant
```bash
$ sog4onnx \
--op_type Constant \
--opset 11 \
--op_name const_custom1 \
--output_variables boxes float32 [1,6,4] \
--attributes value float32 \
[[\
[0.5,0.5,1.0,1.0],\
[0.5,0.6,1.0,1.0],\
[0.5,0.4,1.0,1.0],\
[0.5,10.5,1.0,1.0],\
[0.5,10.6,1.0,1.0],\
[0.5,100.5,1.0,1.0]\
]]
```
![image](https://user-images.githubusercontent.com/33194443/163311192-b559134f-d42d-4119-8990-0f7ac63230e3.png)

### 6-5. opset=11, EfficientNMS_TRT (TensorRT Efficient NMS Plugin)
```bash
$ sog4onnx \
--op_type EfficientNMS_TRT \
--opset 11 \
--op_name trt_nms_efficient_std_11 \
--input_variables boxes float32 [1,3549,4] \
--input_variables scores float32 [1,3549,16] \
--attributes plugin_version str 1 \
--attributes score_threshold float32 0.25 \
--attributes iou_threshold float32 0.45 \
--attributes max_output_boxes int64 20 \
--attributes background_class int64 -1 \
--attributes score_activation bool False \
--attributes box_coding int64 0 \
--output_variables num_detections int32 [1,1] \
--output_variables detection_boxes float32 [1,20,4] \
--output_variables detection_scores float32 [1,20] \
--output_variables detection_classes int32 [1,20]
```
![image](https://github.com/PINTO0309/sog4onnx/assets/33194443/1b3989fd-cd73-4b1e-af59-cda25ea61a97)

## 7. Reference
1. https://github.com/onnx/onnx/blob/main/docs/Operators.md
2. https://docs.nvidia.com/deeplearning/tensorrt/onnx-graphsurgeon/docs/index.html
3. https://github.com/NVIDIA/TensorRT/tree/main/tools/onnx-graphsurgeon
4. https://github.com/PINTO0309/sne4onnx
5. https://github.com/PINTO0309/snd4onnx
6. https://github.com/PINTO0309/snc4onnx
7. https://github.com/PINTO0309/scs4onnx
8. https://github.com/PINTO0309/PINTO_model_zoo

## 8. Issues
https://github.com/PINTO0309/simple-onnx-processing-tools/issues
