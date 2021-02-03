import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'//..//..')
import python.flexflow.onnx.model as model

def test_onnx_init():
	model.ONNXModel("mnist_mlp.onnx")