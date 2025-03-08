#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

from collections import OrderedDict
from .tc import TC_Whitelist
from .utility import Utility
from .base import OperatorLayerBase


class Linear(OperatorLayerBase):
    '''
    Notes:
    - If the bias occurs before the GEMM, then its 1 write (bias expansion).
    - If the bias occurs after, then its 1 read and 1 write.
    - Bias in bprop is a reduction and hence is 1 read.
    '''

    gemmKernels = [
        "gemm", "gemv", "dot_kernel", "splitKreduce_kernel",
        "reduce_1Block_kernel", "cutlass"
    ]

    biasKernels = [
        "kernelReduceContigDim", "kernelReduceNoncontigDim_shared",
        "elementwise_kernel", "reduce_kernel", "kernelPointwiseApply2",
        "2d_grouped_direct_kernel", "enable_if",
        "cublasLt::epilogue::impl::globalKernel", "std::enable_if"
    ]

    def __init__(self, d):
        self.name = d.name
        self.dir = d.dir
        self.sub = d.sub

        marker = eval(d.argMarker[0])
        mod = marker['mod']
        op = marker['op']
        args = marker['args']

        assert (mod == "torch.nn.functional")
        assert (op == "linear")

        self.setXWBMNK(args)

        if any(x in d.name for x in Linear.gemmKernels):
            self.op_ = "linear"
        else:
            if not any(x in d.name for x in Linear.biasKernels):
                print(f"⚠️ Warning: Unknown kernel encountered: {d.name}")
            assert any(x in d.name for x in Linear.biasKernels), f"Kernel name: {d.name}"
            self.op_ = "bias"

    def setXWBMNK(self, args):
        x, w, b = (None, None, None)
        if len(args) == 2:
            x, w = args
        elif len(args) == 3:
            x, w, b = args
            assert (x['type'] == w['type'] == "tensor")
            if b['type'] == "tensor":
                assert len(b['shape']) == 1
            elif b['type'] == "NoneType":
                assert b['value'] is None
                b = None
            else:
                assert False
        else:
            assert False

        assert len(w['shape']) == 2
        k1 = x['shape'][-1]
        n, k2 = w['shape']
        assert k1 == k2
        if b is not None:
            assert b['shape'][0] == n
        t1 = x['dtype']
        t2 = w['dtype']
        assert t1 == t2

        self.x = x['shape']
        self.w = w['shape']
        self.b = b['shape'] if b is not None else None
        self.type = t1

        n = self.x[0:-1]
        k = self.x[-1]
        m, k1 = self.w
        assert k == k1

        self.m = m
        self.n = n
        self.k = k

    def op(self):
        return self.op_

    def bytesFlops(self):
        m, n, k = self.m, Utility.numElems(self.n), self.k

        if self.op_ == "linear":
            f = m * n * k * 2
            b = (m * n + m * k + n * k) * Utility.typeToBytes(self.type)
        elif self.op_ == "bias":
            f = m * n
            b = 2 * m * n * Utility.typeToBytes(self.type)
        else:
            assert False
        return b, f

    def bytes(self):
        b, _ = self.bytesFlops()
        return b

    def flops(self):
        _, f = self.bytesFlops()
        return f
