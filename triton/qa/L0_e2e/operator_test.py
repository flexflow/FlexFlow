#------------------------------------------------------------------------------#
# Copyright 2022 NVIDIA CORPORATION
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#------------------------------------------------------------------------------#

import unittest
import numpy as np
import tritonhttpclient
from tritonclientutils import InferenceServerException
from functools import reduce
import test_helpers as helper

import os
import sys
import argparse
import logging


class OperatorTest(unittest.TestCase):

    def setUp(self):
        self.client = tritonhttpclient.InferenceServerClient(
            url="localhost:8000")

    def test_identity(self):
        log = logging.getLogger("Operator Test Logging")
        log.debug("====== test_identity ======")
        model_name = "identity"

        # Prepare input
        input_shape = [4, 1, 5, 5]
        ec = reduce((lambda x, y: x * y), input_shape)
        input_data = np.arange(ec, dtype=np.float32).reshape(input_shape)
        inputs = [tritonhttpclient.InferInput('input', input_shape, "FP32")]
        inputs[0].set_data_from_numpy(input_data)

        output_name = 'output'
        outputs = [tritonhttpclient.InferRequestedOutput(output_name)]

        log.debug("input data: {}".format(input_data))

        try:
            result = self.client.infer(model_name=model_name,
                                       inputs=inputs,
                                       outputs=outputs)

            # Validate the results by comparing with precomputed values.
            output_data = result.as_numpy(output_name)
            self.assertTrue(
                np.array_equal(output_data, input_data),
                "Expect response to have value {}, got {}".format(
                    input_data, output_data))
            log.debug("output data: {}".format(output_data))
        except InferenceServerException as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))
        log.debug("====== end of test_identity ======")

    def test_add(self):
        log = logging.getLogger("Operator Test Logging")
        log.debug("====== test_add ======")
        model_name = "add"

        # Prepare input
        input_shape = [4, 2]
        ec = reduce((lambda x, y: x * y), input_shape)
        input_data = np.arange(ec, dtype=np.float32).reshape(input_shape)
        inputs = [
            tritonhttpclient.InferInput('input0', input_shape, "FP32"),
            tritonhttpclient.InferInput('input1', input_shape, "FP32")
        ]
        inputs[0].set_data_from_numpy(input_data)
        inputs[1].set_data_from_numpy(input_data)

        output_name = 'output'
        outputs = [tritonhttpclient.InferRequestedOutput(output_name)]
        expected_output_data = input_data + input_data

        log.debug("input data: {}".format(input_data))

        try:
            result = self.client.infer(model_name=model_name,
                                       inputs=inputs,
                                       outputs=outputs)

            # Validate the results by comparing with precomputed values.
            output_data = result.as_numpy(output_name)
            self.assertTrue(
                np.array_equal(output_data, expected_output_data),
                "Expect response to have value {}, got {}".format(
                    expected_output_data, output_data))
            log.debug("output data: {}".format(output_data))
        except InferenceServerException as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))
        log.debug("====== end of test_add ======")

    def test_sub(self):
        log = logging.getLogger("Operator Test Logging")
        log.debug("====== test_sub ======")
        model_name = "sub"

        # Prepare input
        input_shape = [4, 2]
        ec = reduce((lambda x, y: x * y), input_shape)
        input0_data = np.ones(input_shape,
                              dtype=np.float32).reshape(input_shape)
        input1_data = np.arange(ec, dtype=np.float32).reshape(input_shape)
        inputs = [
            tritonhttpclient.InferInput('input0', input_shape, "FP32"),
            tritonhttpclient.InferInput('input1', input_shape, "FP32")
        ]
        inputs[0].set_data_from_numpy(input0_data)
        inputs[1].set_data_from_numpy(input1_data)

        output_name = 'output'
        outputs = [tritonhttpclient.InferRequestedOutput(output_name)]
        expected_output_data = input0_data - input1_data

        log.debug("input0 data: {}".format(input0_data))
        log.debug("input1 data: {}".format(input1_data))

        try:
            result = self.client.infer(model_name=model_name,
                                       inputs=inputs,
                                       outputs=outputs)

            # Validate the results by comparing with precomputed values.
            output_data = result.as_numpy(output_name)
            self.assertTrue(
                np.array_equal(output_data, expected_output_data),
                "Expect response to have value {}, got {}".format(
                    expected_output_data, output_data))
            log.debug("output data: {}".format(output_data))
        except InferenceServerException as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))
        log.debug("====== end of test_sub ======")

    def test_mul(self):
        log = logging.getLogger("Operator Test Logging")
        log.debug("====== test_mul ======")
        model_name = "mul"

        # Prepare input
        input_shape = [4, 2]
        ec = reduce((lambda x, y: x * y), input_shape)
        input_data = np.arange(ec, dtype=np.float32).reshape(input_shape)
        inputs = [
            tritonhttpclient.InferInput('input0', input_shape, "FP32"),
            tritonhttpclient.InferInput('input1', input_shape, "FP32")
        ]
        inputs[0].set_data_from_numpy(input_data)
        inputs[1].set_data_from_numpy(input_data)

        output_name = 'output'
        outputs = [tritonhttpclient.InferRequestedOutput(output_name)]
        expected_output_data = input_data * input_data

        log.debug("input data: {}".format(input_data))

        try:
            result = self.client.infer(model_name=model_name,
                                       inputs=inputs,
                                       outputs=outputs)

            # Validate the results by comparing with precomputed values.
            output_data = result.as_numpy(output_name)
            self.assertTrue(
                np.array_equal(output_data, expected_output_data),
                "Expect response to have value {}, got {}".format(
                    expected_output_data, output_data))
            log.debug("output data: {}".format(output_data))
        except InferenceServerException as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))
        log.debug("====== end of test_mul ======")

    def test_tanh(self):
        log = logging.getLogger("Operator Test Logging")
        log.debug("====== test_tanh ======")
        model_name = "tanh"

        # Prepare input
        input_shape = [3, 1]
        ec = reduce((lambda x, y: x * y), input_shape)
        input_data = np.arange(ec, dtype=np.float32).reshape(input_shape)
        inputs = [tritonhttpclient.InferInput('input', input_shape, "FP32")]
        inputs[0].set_data_from_numpy(input_data)

        output_name = 'output'
        outputs = [tritonhttpclient.InferRequestedOutput(output_name)]

        log.debug("input data: {}".format(input_data))
        expected_output_data = np.tanh(input_data)

        try:
            result = self.client.infer(model_name=model_name,
                                       inputs=inputs,
                                       outputs=outputs)

            # Validate the results by comparing with precomputed values.
            output_data = result.as_numpy(output_name)
            self.assertTrue(
                np.array_equal(output_data, expected_output_data),
                "Expect response to have value {}, got {}".format(
                    expected_output_data, output_data))
            log.debug("output data: {}".format(input_data))
        except InferenceServerException as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))
        log.debug("====== end of test_tanh ======")

    def test_reciprocal(self):
        log = logging.getLogger("Operator Test Logging")
        log.debug("====== test_reciprocal ======")
        model_name = "reciprocal"

        # Prepare input
        input_shape = [1, 3]
        ec = reduce((lambda x, y: x * y), input_shape)
        input_data = np.linspace(0, .1, ec,
                                 dtype=np.float32).reshape(input_shape)
        inputs = [tritonhttpclient.InferInput('input', input_shape, "FP32")]
        inputs[0].set_data_from_numpy(input_data)

        output_name = 'output'
        outputs = [tritonhttpclient.InferRequestedOutput(output_name)]

        log.debug("input data: {}".format(input_data))
        expected_output_data = np.reciprocal(input_data)

        try:
            result = self.client.infer(model_name=model_name,
                                       inputs=inputs,
                                       outputs=outputs)

            # Validate the results by comparing with precomputed values.
            output_data = result.as_numpy(output_name)
            self.assertTrue(
                np.array_equal(output_data, expected_output_data),
                "Expect response to have value {}, got {}".format(
                    expected_output_data, output_data))
            log.debug("output data: {}".format(input_data))
        except InferenceServerException as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))
        log.debug("====== end of test_reciprocal ======")

    def test_sqrt(self):
        log = logging.getLogger("Operator Test Logging")
        log.debug("====== test_sqrt ======")
        model_name = "sqrt"

        # Prepare input
        input_shape = [3, 1]
        ec = reduce((lambda x, y: x * y), input_shape)
        input_data = np.arange(ec, dtype=np.float32).reshape(input_shape)
        inputs = [tritonhttpclient.InferInput('input', input_shape, "FP32")]
        inputs[0].set_data_from_numpy(input_data)

        output_name = 'output'
        outputs = [tritonhttpclient.InferRequestedOutput(output_name)]

        log.debug("input data: {}".format(input_data))
        expected_output_data = np.sqrt(input_data)

        try:
            result = self.client.infer(model_name=model_name,
                                       inputs=inputs,
                                       outputs=outputs)

            # Validate the results by comparing with precomputed values.
            output_data = result.as_numpy(output_name)
            self.assertTrue(
                np.array_equal(output_data, expected_output_data),
                "Expect response to have value {}, got {}".format(
                    expected_output_data, output_data))
            log.debug("output data: {}".format(input_data))
        except InferenceServerException as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))
        log.debug("====== end of test_sqrt ======")

    def test_cast(self):
        log = logging.getLogger("Operator Test Logging")
        log.debug("====== test_cast ======")
        model_name = "cast"

        # Prepare input
        input_shape = [1, 3]
        ec = reduce((lambda x, y: x * y), input_shape)
        input_data = np.linspace(0, .1, ec,
                                 dtype=np.float32).reshape(input_shape)
        inputs = [tritonhttpclient.InferInput('input', input_shape, "FP32")]
        inputs[0].set_data_from_numpy(input_data)

        output_name = 'output'
        outputs = [tritonhttpclient.InferRequestedOutput(output_name)]

        log.debug("input data: {}".format(input_data))
        expected_output_data = input_data.astype(np.double, copy=True)

        try:
            result = self.client.infer(model_name=model_name,
                                       inputs=inputs,
                                       outputs=outputs)

            # Validate the results by comparing with precomputed values.
            output_data = result.as_numpy(output_name)
            self.assertTrue(
                np.array_equal(output_data, expected_output_data),
                "Expect response to have value {}, got {}".format(
                    expected_output_data, output_data))
            log.debug("output data: {}".format(input_data))
        except InferenceServerException as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))
        log.debug("====== end of test_cast ======")

    def test_softmax(self):
        log = logging.getLogger("Operator Test Logging")
        log.debug("====== test_softmax ======")
        model_name = "softmax"

        # Prepare input
        input_shape = [3, 1]
        ec = reduce((lambda x, y: x * y), input_shape)
        input_data = np.arange(ec, dtype=np.float32).reshape(input_shape)
        inputs = [tritonhttpclient.InferInput('input', input_shape, "FP32")]
        inputs[0].set_data_from_numpy(input_data)

        output_name = 'output'
        outputs = [tritonhttpclient.InferRequestedOutput(output_name)]

        log.debug("input data: {}".format(input_data))
        expected_output_data = helper.softmax(input_data, 0)

        try:
            result = self.client.infer(model_name=model_name,
                                       inputs=inputs,
                                       outputs=outputs)

            # Validate the results by comparing with precomputed values.
            output_data = result.as_numpy(output_name)
            self.assertTrue(
                np.allclose(output_data, expected_output_data, atol=1e-07),
                "Expect response to have value {}, got {}".format(
                    expected_output_data, output_data))
            log.debug("output data: {}".format(input_data))
        except InferenceServerException as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))
        log.debug("====== end of test_softmax ======")

    def test_softmax_default_axis(self):
        log = logging.getLogger("Operator Test Logging")
        log.debug("====== test_softmax_default_axis ======")
        model_name = "softmax1"

        # Prepare input
        input_shape = [3, 1]
        ec = reduce((lambda x, y: x * y), input_shape)
        input_data = np.arange(ec, dtype=np.float32).reshape(input_shape)
        inputs = [tritonhttpclient.InferInput('input', input_shape, "FP32")]
        inputs[0].set_data_from_numpy(input_data)

        output_name = 'output'
        outputs = [tritonhttpclient.InferRequestedOutput(output_name)]

        log.debug("input data: {}".format(input_data))
        expected_output_data = helper.softmax(input_data, 1)

        try:
            result = self.client.infer(model_name=model_name,
                                       inputs=inputs,
                                       outputs=outputs)

            # Validate the results by comparing with precomputed values.
            output_data = result.as_numpy(output_name)
            self.assertTrue(
                np.allclose(output_data, expected_output_data, atol=1e-07),
                "Expect response to have value {}, got {}".format(
                    expected_output_data, output_data))
            log.debug("output data: {}".format(input_data))
        except InferenceServerException as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))
        log.debug("====== end of test_softmax_default_axis ======")


if __name__ == '__main__':
    if 'LEGION_BACKEND_TEST_LOG_LEVEL' in os.environ:
        level_config = {'debug': logging.DEBUG, 'info': logging.INFO}
        logging.basicConfig(stream=sys.stderr)
        log_level = level_config[
            os.environ['LEGION_BACKEND_TEST_LOG_LEVEL'].lower()]
        logging.getLogger("Operator Test Logging").setLevel(log_level)

    unittest.main()
