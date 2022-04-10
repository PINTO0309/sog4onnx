# [WIP] sog4onnx
Simple ONNX operation generator. **S**imple **O**peration **G**enerator for **ONNX**.

# Key concept

- [ ] Variable, Constant, Operation and Attribute can be generated externally.
- [ ] Allow Opset to be specified externally.
- [ ] No check for consistency of Operations within the tool, as new OPs are added frequently and the definitions of existing OPs change with each new version of ONNX's Opset.
- [ ] Only one OP can be defined at a time, and the goal is to generate free ONNX graphs using a combination of **[snc4onnx](https://github.com/PINTO0309/snc4onnx)**, **[sne4onnx](https://github.com/PINTO0309/sne4onnx)**, **[snd4onnx](https://github.com/PINTO0309/snd4onnx)** and **[scs4onnx](https://github.com/PINTO0309/scs4onnx)**.

## 7. Reference
1. https://docs.nvidia.com/deeplearning/tensorrt/onnx-graphsurgeon/docs/index.html
2. https://github.com/NVIDIA/TensorRT/tree/main/tools/onnx-graphsurgeon
3. https://github.com/PINTO0309/sne4onnx
4. https://github.com/PINTO0309/snd4onnx
5. https://github.com/PINTO0309/snc4onnx
6. https://github.com/PINTO0309/scs4onnx
7. https://github.com/PINTO0309/PINTO_model_zoo
