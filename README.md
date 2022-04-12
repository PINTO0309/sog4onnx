# [WIP] sog4onnx
Simple ONNX operation generator. **S**imple **O**peration **G**enerator for **ONNX**.

https://github.com/PINTO0309/simple-onnx-processing-tools

# Key concept

- [x] Variable, Constant, Operation and Attribute can be generated externally.
- [x] Allow Opset to be specified externally.
- [x] No check for consistency of Operations within the tool, as new OPs are added frequently and the definitions of existing OPs change with each new version of ONNX's Opset.
- [x] Only one OP can be defined at a time, and the goal is to generate free ONNX graphs using a combination of **[snc4onnx](https://github.com/PINTO0309/snc4onnx)**, **[sne4onnx](https://github.com/PINTO0309/sne4onnx)**, **[snd4onnx](https://github.com/PINTO0309/snd4onnx)** and **[scs4onnx](https://github.com/PINTO0309/scs4onnx)**.

## 6. Sample
```bash
$ sog4onnx \
--op_type Gemm \
--opset 11 \
--input_variables i1 float32 [1,2,3] \
--input_variables i2 float32 [1,1] \
--input_variables i3 int32 [0] \
--output_variables o1 float32 [1,2,3] \
--attributes "{\"alpha\": 1.0, \"beta\": 1.0, \"broadcast\": 0, \"transA\": 0, \"transB\": 0}" \
--non_verbose
```
![image](https://user-images.githubusercontent.com/33194443/163012020-0ee8e0f9-be9d-4954-b080-6c2762ce54e7.png)

## 7. Reference
1. https://docs.nvidia.com/deeplearning/tensorrt/onnx-graphsurgeon/docs/index.html
2. https://github.com/NVIDIA/TensorRT/tree/main/tools/onnx-graphsurgeon
3. https://github.com/PINTO0309/sne4onnx
4. https://github.com/PINTO0309/snd4onnx
5. https://github.com/PINTO0309/snc4onnx
6. https://github.com/PINTO0309/scs4onnx
7. https://github.com/PINTO0309/PINTO_model_zoo
