#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import numpy as np

import paddle
from paddle import fluid
from paddle.jit import to_static


@paddle.jit.to_static
def dygraph_decorated_func(x):
    x = fluid.dygraph.to_variable(x)
    if paddle.mean(x) > 0:
        x_v = x - 1
    else:
        x_v = x + 1
    return x_v


@paddle.jit.to_static
def jit_decorated_func(x):
    x = fluid.dygraph.to_variable(x)
    if paddle.mean(x) > 0:
        x_v = x - 1
    else:
        x_v = x + 1
    return x_v


@paddle.jit.to_static
def decorated_call_decorated(x):
    return jit_decorated_func(x)


class DoubleDecorated:
    @classmethod
    @to_static
    def double_decorated_func1(self, x):
        return dygraph_decorated_func(x)

    @classmethod
    @paddle.jit.to_static
    def double_decorated_func2(self, x):
        return jit_decorated_func(x)


class TestFullNameDecorator(unittest.TestCase):
    def test_run_success(self):
        x = np.ones([1, 2]).astype("float32")
        answer = np.zeros([1, 2]).astype("float32")
        with fluid.dygraph.guard():
            np.testing.assert_allclose(
                dygraph_decorated_func(x).numpy(), answer, rtol=1e-05
            )
            np.testing.assert_allclose(
                jit_decorated_func(x).numpy(), answer, rtol=1e-05
            )
            np.testing.assert_allclose(
                decorated_call_decorated(x).numpy(), answer, rtol=1e-05
            )
            with self.assertRaises(NotImplementedError):
                DoubleDecorated().double_decorated_func1(x)
            with self.assertRaises(NotImplementedError):
                DoubleDecorated().double_decorated_func2(x)


if __name__ == '__main__':
    unittest.main()
