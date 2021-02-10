import pytest
import flexflow.onnx.model as model

def test_onnx_init():
	model.ONNXModel("mnist_mlp_pt.onnx")