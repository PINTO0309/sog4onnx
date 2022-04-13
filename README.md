# sog4onnx
Simple ONNX operation generator. **S**imple **O**peration **G**enerator for **ONNX**.

https://github.com/PINTO0309/simple-onnx-processing-tools

[![Downloads](https://static.pepy.tech/personalized-badge/sog4onnx?period=total&units=none&left_color=grey&right_color=brightgreen&left_text=Downloads)](https://pepy.tech/project/sog4onnx) ![GitHub](https://img.shields.io/github/license/PINTO0309/sog4onnx?color=2BAF2B) [![PyPI](https://img.shields.io/pypi/v/sog4onnx?color=2BAF2B)](https://pypi.org/project/sog4onnx/) [![CodeQL](https://github.com/PINTO0309/sog4onnx/workflows/CodeQL/badge.svg)](https://github.com/PINTO0309/sog4onnx/actions?query=workflow%3ACodeQL)

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
```bash
### docker pull
$ docker pull pinto0309/sog4onnx:latest

### docker build
$ docker build -t pinto0309/sog4onnx:latest .

### docker run
$ docker run --rm -it -v `pwd`:/workdir pinto0309/sog4onnx:latest
$ cd /workdir
```

## 2. CLI Usage
```bash
$ sog4onnx -h

usage: sog4onnx [-h]
  --op_type OP_TYPE
  --opset OPSET
  [--input_variables NAME TYPE VALUE]
  [--output_variables NAME TYPE VALUE]
  [--attributes NAME VALUE]
  [--output_onnx_file_path OUTPUT_ONNX_FILE_PATH]
  [--non_verbose]

optional arguments:
  -h, --help
        show this help message and exit

  --op_type OP_TYPE
        ONNX OP type.
        https://github.com/onnx/onnx/blob/main/docs/Operators.md

  --opset OPSET
        ONNX opset number.

  --input_variables NAME TYPE VALUE
        input_variables can be specified multiple times.
        --input_variables variable_name numpy.dtype shape
        https://github.com/onnx/onnx/blob/main/docs/Operators.md

        e.g.
        --input_variables i1 float32 [1,3,5,5] \
        --input_variables i2 int32 [1] \
        --input_variables i3 float64 [1,3,224,224]

  --output_variables NAME TYPE VALUE
        output_variables can be specified multiple times.
        --output_variables variable_name numpy.dtype shape
        https://github.com/onnx/onnx/blob/main/docs/Operators.md

        e.g.
        --output_variables o1 float32 [1,3,5,5] \
        --output_variables o2 int32 [1] \
        --output_variables o3 float64 [1,3,224,224]

  --attributes NAME VALUE
        attributes can be specified multiple times.
        --attributes name value
        https://github.com/onnx/onnx/blob/main/docs/Operators.md

        e.g.
        --attributes alpha 1.0 \
        --attributes beta 1.0 \
        --attributes transA 0 \
        --attributes transB 0

  --output_onnx_file_path OUTPUT_ONNX_FILE_PATH
        Output onnx file path. If not specified, a file with the OP type name is generated.

        e.g. op_type="Gemm" -> Gemm.onnx

  --non_verbose
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

    input_variables: Optional[dict]
        Specify input variables for the OP to be generated.
        See below for the variables that can be specified.
        https://github.com/onnx/onnx/blob/main/docs/Operators.md
        {'input_var_name1': [numpy.dtype, shape], 'input_var_name2': [dtype, shape], ...}

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
        {'output_var_name1': [numpy.dtype, shape], 'output_var_name2': [dtype, shape], ...}

        e.g.
        output_variables = {
          "name1": [np.float32, [1,224,224,3]],
          "name2": [np.bool_, [0]],
          ...
        }

    attributes: Optional[dict]
        Specify output attributes for the OP to be generated.
        See below for the attributes that can be specified.
        https://github.com/onnx/onnx/blob/main/docs/Operators.md
        {'attr_name1': value1, 'attr_name2': value2, 'attr_name3': value3, ...}

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
--input_variables i1 float32 [1,2,3] \
--input_variables i2 float32 [1,1] \
--input_variables i3 int32 [0] \
--output_variables o1 float32 [1,2,3] \
--attributes alpha 1.0 \
--attributes beta 1.0 \
--attributes transA 0 \
--attributes transB 0
```

## 5. In-script Execution
```python
from sog4onnx import generate

single_op_graph = generate(
    op_type = 'Gemm',
    opset = 1,
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
--input_variables i1 float32 [1,2,3] \
--input_variables i2 float32 [1,1] \
--input_variables i3 int32 [0] \
--output_variables o1 float32 [1,2,3] \
--attributes alpha 1.0 \
--attributes beta 1.0 \
--attributes transA 0 \
--attributes transB 0
--non_verbose
```
![image](https://user-images.githubusercontent.com/33194443/163018526-f2d5c647-c3e9-4e65-9b9a-c1c4fa5da8a5.png)
![image](https://user-images.githubusercontent.com/33194443/163018647-a6880370-8772-4af1-9ffe-59820a621c30.png)

### 6-2. opset=11, Add
```bash
$ sog4onnx \
--op_type Add \
--opset 11 \
--input_variables i1 float32 [1,2,3] \
--input_variables i2 float32 [1,2,3] \
--output_variables o1 float32 [1,2,3] \
--non_verbose
```
![image](https://user-images.githubusercontent.com/33194443/163042479-9998ba73-ee26-44ea-bd6b-dcd04539190b.png)
![image](https://user-images.githubusercontent.com/33194443/163042529-5dbd1b5f-e8d1-47d0-8a9e-aacd91539c2b.png)

## 7. Reference
1. https://github.com/onnx/onnx/blob/main/docs/Operators.md
2. https://docs.nvidia.com/deeplearning/tensorrt/onnx-graphsurgeon/docs/index.html
3. https://github.com/NVIDIA/TensorRT/tree/main/tools/onnx-graphsurgeon
4. https://github.com/PINTO0309/sne4onnx
5. https://github.com/PINTO0309/snd4onnx
6. https://github.com/PINTO0309/snc4onnx
7. https://github.com/PINTO0309/scs4onnx
8. https://github.com/PINTO0309/PINTO_model_zoo
