import unittest

import torch

from mnist_gan.modules import Discriminator, Generator, MaxoutLayer


class TestMNISTGenerator(unittest.TestCase):
    def setUp(self):
        self.generator = Generator(
            num_dimensions_z=100,
            num_classes=10,
            num_dimensions_output=784,
        )

    def test_feedforward_one_sample(self):
        z = torch.rand(100)
        condition_vector = torch.tensor(
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            dtype=torch.float32
        )
        y_hat = self.generator(z, condition_vector)

        self.assertEqual(y_hat.size(0), 784)
        self.assertLessEqual(y_hat.max(), 1)
        self.assertGreaterEqual(y_hat.min(), 0)

    def test_feedforward_minibatch(self):
        z = torch.rand(4, 100)
        condition_vector = torch.tensor(
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            ], dtype=torch.float32)
        y_hat = self.generator(z, condition_vector)

        self.assertEqual(y_hat.size(0), 4)
        self.assertEqual(y_hat.size(1), 784)
        self.assertLessEqual(y_hat.max(), 1)
        self.assertGreaterEqual(y_hat.min(), 0)


class TestMaxoutLayer(unittest.TestCase):
    def test_feedforward(self):
        maxout = MaxoutLayer(
            num_pieces=5,
            num_input_dimensions=40,
            num_units=2,
        )
        input = torch.rand(4, 40)
        x = maxout(input)

        self.assertEqual(x.size(0), 4)
        self.assertEqual(x.size(1), 2)


class TestDiscriminator(unittest.TestCase):
    def test_feedforward(self):
        discriminator = Discriminator(
            num_dimensions_x=32,
            num_dimensions_y=4,
        )
        x = torch.rand(4, 32)
        y = torch.tensor([
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
        ], dtype=torch.float)
        is_real = discriminator(x, y)
        self.assertEqual(is_real.size(0), 4)
        self.assertEqual(is_real.size(1), 1)
