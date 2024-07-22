import unittest
import math
import torch
import torch.nn as nn
from xonv.layer import Xonv2D


class TestXonv2D(unittest.TestCase):

    def setUp(self):
        self.batch_size = 2
        self.in_channels = 3
        self.out_channels = 16
        self.kernel_size = 3
        self.input_size = (32, 32)
        self.stride = 1  # Default stride

        self.input_tensor = torch.randn(
            self.batch_size,
            self.in_channels,
            *self.input_size,
        )

        self.standard_conv = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            padding=self.kernel_size // 2,
            bias=True,
            stride=self.stride,
        )
        self.xonv = Xonv2D(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.input_size,
            stride=self.stride,
        )

    def test_output_shape(self):
        """Test if the output shape is correct."""
        output = self.xonv(self.input_tensor)
        expected_shape = (self.batch_size, self.out_channels, *self.input_size)
        self.assertEqual(output.shape, expected_shape)

    def test_same_kernels(self):
        """Test if Xonv2D produces the same output as nn.Conv2d when all kernels
        are the same."""
        with torch.no_grad():
            self.xonv.weights.data.copy_(
                self.standard_conv.weight.data.unsqueeze(0).unsqueeze(
                    0).expand(
                        self.input_size[0],
                        self.input_size[1],
                        -1,
                        -1,
                        -1,
                        -1,
                    ))
            self.xonv.bias.data.copy_(
                self.standard_conv.bias.data.unsqueeze(0).unsqueeze(0).expand(
                    self.input_size[0],
                    self.input_size[1],
                    -1,
                ))

        standard_output = self.standard_conv(self.input_tensor)
        xonv_output = self.xonv(self.input_tensor)

        self.assertTrue(
            torch.allclose(
                standard_output,
                xonv_output,
                atol=1e-6,
            ))

    def test_different_kernels(self):
        """Test if Xonv2D produces different outputs for different spatial
        locations."""
        output1 = self.xonv(self.input_tensor)

        # Modify weights at a specific location
        with torch.no_grad():
            self.xonv.weights.data[0, 0] *= 2

        output2 = self.xonv(self.input_tensor)

        self.assertFalse(torch.allclose(output1, output2))

    def test_backpropagation(self):
        """Test if backpropagation works correctly."""
        self.xonv.zero_grad()
        output = self.xonv(self.input_tensor)
        loss = output.sum()
        loss.backward()

        self.assertIsNotNone(self.xonv.weights.grad)
        self.assertIsNotNone(self.xonv.bias.grad)
        self.assertFalse(
            torch.allclose(
                self.xonv.weights.grad,
                torch.zeros_like(self.xonv.weights.grad),
            ))
        self.assertFalse(
            torch.allclose(
                self.xonv.bias.grad,
                torch.zeros_like(self.xonv.bias.grad),
            ))

    def test_large_kernel(self):
        """Test if the layer works with a larger kernel size."""
        large_kernel_size = 5
        large_kernel_xonv = Xonv2D(
            self.in_channels,
            self.out_channels,
            large_kernel_size,
            self.input_size,
        )
        output = large_kernel_xonv(self.input_tensor)
        expected_shape = (self.batch_size, self.out_channels, *self.input_size)
        self.assertEqual(output.shape, expected_shape)

    def test_different_input_size(self):
        """Test if the layer works with a different input size."""
        different_input_size = (64, 64)
        different_input_tensor = torch.randn(
            self.batch_size,
            self.in_channels,
            *different_input_size,
        )
        different_xonv = Xonv2D(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            different_input_size,
        )
        output = different_xonv(different_input_tensor)
        expected_shape = (
            self.batch_size,
            self.out_channels,
            *different_input_size,
        )
        self.assertEqual(output.shape, expected_shape)

    def test_stride_2(self):
        """Test if the layer works with stride 2."""
        stride = 2
        xonv_stride2 = Xonv2D(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.input_size,
            stride=stride,
        )
        output = xonv_stride2(self.input_tensor)
        expected_shape = (self.batch_size, self.out_channels,
                          self.input_size[0] // stride,
                          self.input_size[1] // stride)
        self.assertEqual(output.shape, expected_shape)

    def test_stride_3(self):
        """Test if the layer works with stride 3."""
        stride = 3
        xonv_stride3 = Xonv2D(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.input_size,
            stride=stride,
        )
        output = xonv_stride3(self.input_tensor)
        expected_shape = (self.batch_size, self.out_channels,
                          math.ceil(self.input_size[0] / stride),
                          math.ceil(self.input_size[1] / stride))
        self.assertEqual(output.shape, expected_shape)

    def test_stride_same_kernels(self):
        """Test if Xonv2D with stride produces the same output as nn.Conv2d with
        stride when all kernels are the same."""
        stride = 2
        standard_conv_stride = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            padding=self.kernel_size // 2,
            bias=True,
            stride=stride,
        )
        xonv_stride = Xonv2D(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.input_size,
            stride=stride,
        )

        with torch.no_grad():
            xonv_stride.weights.data.copy_(
                standard_conv_stride.weight.data.unsqueeze(0).unsqueeze(
                    0).expand(
                        math.ceil(self.input_size[0] / stride),
                        math.ceil(self.input_size[1] / stride),
                        -1,
                        -1,
                        -1,
                        -1,
                    ))
            xonv_stride.bias.data.copy_(
                standard_conv_stride.bias.data.unsqueeze(0).unsqueeze(
                    0).expand(
                        math.ceil(self.input_size[0] / stride),
                        math.ceil(self.input_size[1] / stride),
                        -1,
                    ))

        standard_output = standard_conv_stride(self.input_tensor)
        xonv_output = xonv_stride(self.input_tensor)

        self.assertTrue(
            torch.allclose(
                standard_output,
                xonv_output,
                atol=1e-6,
            ))

    def test_non_divisible_input_size(self):
        """Test if the layer works with an input size not divisible by stride."""
        stride = 3
        input_size = (31, 31)  # Not divisible by 3
        input_tensor = torch.randn(
            self.batch_size,
            self.in_channels,
            *input_size,
        )
        xonv_stride3 = Xonv2D(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            input_size,
            stride=stride,
        )
        output = xonv_stride3(input_tensor)
        expected_shape = (self.batch_size, self.out_channels,
                          math.ceil(input_size[0] / stride),
                          math.ceil(input_size[1] / stride))
        self.assertEqual(output.shape, expected_shape)


if __name__ == '__main__':
    unittest.main()
