import unittest
import numpy as np
from heroch import Tensor
from heroch.nn.linear import Linear
from heroch.nn.activations import ReLU, Sigmoid
from heroch.nn.loss import MSELoss
from heroch.optim.sgd import SGD

class TestHeroch(unittest.TestCase):
    def test_tensor_ops(self):
        a = Tensor([1, 2, 3])
        b = Tensor([4, 5, 6])
        c = a + b
        np.testing.assert_array_equal(c.data, [5, 7, 9])
        
        d = a * b
        np.testing.assert_array_equal(d.data, [4, 10, 18])
        
        e = a - b
        np.testing.assert_array_equal(e.data, [-3, -3, -3])

    def test_autograd_complex(self):
        x = Tensor(2.0, requires_grad=True)
        y = Tensor(3.0, requires_grad=True)
        
        # f(x, y) = x^2 + 3xy
        # df/dx = 2x + 3y = 4 + 9 = 13
        # df/dy = 3x = 6
        
        f = x * x + Tensor(3.0) * x * y
        f.backward()
        
        self.assertEqual(x.grad, 13.0)
        self.assertEqual(y.grad, 6.0)

    def test_linear_layer(self):
        layer = Linear(3, 2)
        x = Tensor([[1, 2, 3]], requires_grad=True)
        out = layer(x)
        self.assertEqual(out.shape, (1, 2))
        
        out.sum().backward()
        self.assertIsNotNone(layer.weight.grad)
        self.assertIsNotNone(layer.bias.grad)
        self.assertIsNotNone(x.grad)

    def test_relu(self):
        relu = ReLU()
        x = Tensor([-1, 0, 1], requires_grad=True)
        out = relu(x)
        np.testing.assert_array_equal(out.data, [0, 0, 1])
        
        out.sum().backward()
        np.testing.assert_array_equal(x.grad, [0, 0, 1])

    def test_sigmoid(self):
        sigmoid = Sigmoid()
        x = Tensor(0.0, requires_grad=True)
        out = sigmoid(x)
        self.assertAlmostEqual(out.data.item(), 0.5)
        
        out.backward()
        # dSigmoid(0)/dx = 0.5 * (1 - 0.5) = 0.25
        self.assertAlmostEqual(x.grad.item(), 0.25)

if __name__ == '__main__':
    unittest.main()
