import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession("backend/app/models/rfdetr-nano.onnx", providers=["CPUExecutionProvider"])
inp = sess.get_inputs()[0].name

x = np.zeros((1,3,384,384), dtype=np.float32)
out0, out1 = sess.run(None, {inp: x})
print("out0 boxes:", out0.shape, out0.dtype)
print("out1 logits:", out1.shape, out1.dtype)
print("boxes min/max:", out0.min(), out0.max())
print("logits min/max:", out1.min(), out1.max())