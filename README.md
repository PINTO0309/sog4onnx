# [WIP] sog4onnx
Simple ONNX operation generator. **S**imple **O**peration **G**enerator for **ONNX**.

https://github.com/PINTO0309/simple-onnx-processing-tools

[![Downloads](https://static.pepy.tech/personalized-badge/sog4onnx?period=total&units=none&left_color=grey&right_color=brightgreen&left_text=Downloads)](https://pepy.tech/project/sog4onnx) ![GitHub](https://img.shields.io/github/license/PINTO0309/sog4onnx?color=2BAF2B) [![PyPI](https://img.shields.io/pypi/v/sog4onnx?color=2BAF2B)](https://pypi.org/project/sog4onnx/) [![CodeQL](https://github.com/PINTO0309/sog4onnx/workflows/CodeQL/badge.svg)](https://github.com/PINTO0309/sog4onnx/actions?query=workflow%3ACodeQL)

# Key concept

- [x] Variable, Constant, Operation and Attribute can be generated externally.
- [x] Allow Opset to be specified externally.
- [x] No check for consistency of Operations within the tool, as new OPs are added frequently and the definitions of existing OPs change with each new version of ONNX's Opset.
- [x] Only one OP can be defined at a time, and the goal is to generate free ONNX graphs using a combination of **[snc4onnx](https://github.com/PINTO0309/snc4onnx)**, **[sne4onnx](https://github.com/PINTO0309/sne4onnx)**, **[snd4onnx](https://github.com/PINTO0309/snd4onnx)** and **[scs4onnx](https://github.com/PINTO0309/scs4onnx)**.
- [x] List of parameters that can be specified: https://github.com/onnx/onnx/blob/main/docs/Operators.md

## 6. Sample
```bash
$ sog4onnx \
--op_type Gemm \
--opset 1 \
--input_variables i1 float32 [1,2,3] \
--input_variables i2 float32 [1,1] \
--input_variables i3 int32 [0] \
--output_variables o1 float32 [1,2,3] \
--attributes "{\"alpha\": 1.0, \"beta\": 1.0, \"broadcast\": 0, \"transA\": 0, \"transB\": 0}" \
--non_verbose
```
![image](https://user-images.githubusercontent.com/33194443/163018526-f2d5c647-c3e9-4e65-9b9a-c1c4fa5da8a5.png)
![image](https://user-images.githubusercontent.com/33194443/163018647-a6880370-8772-4af1-9ffe-59820a621c30.png)

## 7. Reference
1. https://docs.nvidia.com/deeplearning/tensorrt/onnx-graphsurgeon/docs/index.html
2. https://github.com/NVIDIA/TensorRT/tree/main/tools/onnx-graphsurgeon
3. https://github.com/PINTO0309/sne4onnx
4. https://github.com/PINTO0309/snd4onnx
5. https://github.com/PINTO0309/snc4onnx
6. https://github.com/PINTO0309/scs4onnx
7. https://github.com/PINTO0309/PINTO_model_zoo
