import torch
import torch.nn as nn
import pytorch_lightning as pl

from mnist_gan.types import *


class Generator(nn.Module):
    def __init__(
        self,
        num_dimensions_z: int = 100,
        num_classes: int = 10,
        num_dimensions_output: int = 784,
    ) -> None:
        super(Generator, self).__init__()
        self.num_dimensions_z = num_dimensions_z
        self.num_classes = num_classes
        self.num_dimensions_output = num_dimensions_output

        self.linear_noise = nn.Sequential(
            nn.Linear(
                in_features=self.num_dimensions_z,
                out_features=200,
            ),
            nn.ReLU(),
        )
        self.linear_condition = nn.Sequential(
            nn.Linear(
                in_features=self.num_classes,
                out_features=1000,
            ),
            nn.ReLU(),
        )
        self.linear_output = nn.Sequential(
            nn.Linear(
                in_features=1200,
                out_features=self.num_dimensions_output,
            ),
            nn.Sigmoid()
        )

        self.noise_distribution = torch.distributions.Uniform(low=-1 / 2, high=1 / 2)

    def forward(
        self,
        z: torch.Tensor,
        condition_vector: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args
        z: A noise prior z with dimensionality 100 drawn from a uniform distribution within the unit hypercube.
        condition_vector: The class labels, encoded as one-hot vectors.
        """

        z = self.linear_noise(z)
        condition_vector = self.linear_condition(condition_vector)

        x = torch.cat([z, condition_vector], dim=-1)
        x = self.linear_output(x)

        return x

    def sample_noise(self, batch_size) -> torch.Tensor:
        z = self.noise_distribution.sample(torch.Size([batch_size, self.num_dimensions_z]))
        return z


class MaxoutLayer(nn.Module):
    def __init__(
        self,
        num_pieces: int,
        num_input_dimensions: int,
        num_units: int,
    ):
        super(MaxoutLayer, self).__init__()
        self.Ws = nn.Parameter(
            data=torch.rand(num_pieces, num_input_dimensions, num_units)
        )
        self.bs = nn.Parameter(
            data=torch.rand(num_pieces, num_units)
        )
        torch.nn.init.uniform_(self.Ws, a=-1 / num_input_dimensions, b=1 / num_input_dimensions)
        torch.nn.init.uniform_(self.bs, a=-1 / num_input_dimensions, b=1 / num_input_dimensions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.einsum('pio,bi->bpo', self.Ws, x)
        x = x + self.bs
        x = x.max(1)[0]

        return x


class Discriminator(nn.Module):
    def __init__(
        self,
        num_dimensions_x: int = 784,
        num_dimensions_y: int = 10,
    ) -> None:
        super(Discriminator, self).__init__()
        self.maxout_x = MaxoutLayer(
            num_pieces=5,
            num_input_dimensions=num_dimensions_x,
            num_units=240,
        )
        self.maxout_y = MaxoutLayer(
            num_pieces=5,
            num_input_dimensions=num_dimensions_y,
            num_units=50,
        )
        self.maxout_joint = MaxoutLayer(
            num_pieces=5,
            num_input_dimensions=290,
            num_units=240,
        )
        self.linear_fc = nn.Linear(
            in_features=240,
            out_features=1,
        )

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        x = self.maxout_x(x)
        y = self.maxout_y(y)
        x = torch.cat([x, y], dim=-1)
        x = self.maxout_joint(x)
        x = self.linear_fc(x)
        x = x.sigmoid()
        return x


class GAN(pl.LightningModule):
    def __init__(
        self,
        lr: float = 0.1,
        *args, **kwargs,
    ) -> None:
        super(GAN, self).__init__()
        self.generator = Generator()
        self.discriminator = Discriminator()

        self.lr = lr

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, label = batch
        x = x.flatten(1)
        z = self.generator.sample_noise(
            batch_size=x.size(0),
        )
        if optimizer_idx == 0:  # Train generator
            x_hat = self.generator(z, label)
            is_real_for_fake = self.discriminator(x_hat, label)
            loss_generator = is_real_for_fake.add(1e-8).log().neg().mean()
            return {'loss': loss_generator, 'optimizer_idx': optimizer_idx}

        if optimizer_idx == 1:
            x_hat = self.generator(z, label)
            is_fake_for_fake = self.discriminator(x_hat, label)
            is_real_for_real = self.discriminator(x, label)
            loss_discriminator = is_fake_for_fake.add(1e-8).log().neg().mean() + is_real_for_real.add(1e-8).log().neg().mean()
            loss_discriminator = loss_discriminator / 2
            return {'loss': loss_discriminator, 'optimizer_idx': optimizer_idx}

    def on_train_batch_end(self, outputs, batch, batch_idx):
        for output in outputs:
            if output['optimizer_idx'] == 0:
                self.log(
                    name='Loss/Generator/Train',
                    value=output['loss'],
                )
            elif output['optimizer_idx'] == 1:
                self.log(
                    name='Loss/Discriminator/Train',
                    value=output['loss'],
                )
            else:
                raise ValueError

    def validation_step(self, batch, batch_idx):
        x, label = batch
        x = x.flatten(1)
        z = self.generator.sample_noise(
            batch_size=x.size(0),
        )
        x_hat = self.generator(z, label)
        is_real_for_fake = self.discriminator(x_hat, label)
        loss_generator = is_real_for_fake.add(1e-8).log().neg().mean()

        x_hat = self.generator(z, label)
        is_fake_for_fake = self.discriminator(x_hat, label)
        is_real_for_real = self.discriminator(x, label)
        loss_discriminator = is_fake_for_fake.add(1e-8).log().neg().mean() + is_real_for_real.add(1e-8).log().neg().mean()
        loss_discriminator = loss_discriminator / 2

        return {'loss_generator': loss_generator, 'loss_discriminator': loss_discriminator}

    def validation_step_end(self, outputs):
        self.log(
            name='Loss/Generator/Valid',
            value=outputs['loss_generator'],
        )
        self.log(
            name='Loss/Discriminator/Valid',
            value=outputs['loss_discriminator'],
        )

    def test_step(self, batch, batch_idx):
        x, label = batch
        x = x.flatten(1)
        z = self.generator.sample_noise(
            batch_size=x.size(0),
        )
        x_hat = self.generator(z, label)
        is_real_for_fake = self.discriminator(x_hat, label)
        loss_generator = is_real_for_fake.add(1e-8).log().neg().mean()

        x_hat = self.generator(z, label)
        is_fake_for_fake = self.discriminator(x_hat, label)
        is_real_for_real = self.discriminator(x, label)
        loss_discriminator = is_fake_for_fake.add(1e-8).log().neg().mean() + is_real_for_real.add(1e-8).log().neg().mean()
        loss_discriminator = loss_discriminator / 2

        return {'loss_generator': loss_generator, 'loss_discriminator': loss_discriminator}

    def configure_optimizers(self):
        optimizer_generator = torch.optim.SGD(
            params=self.generator.parameters(),
            lr=self.lr,
        )
        optimizer_discriminator = torch.optim.SGD(
            params=self.discriminator.parameters(),
            lr=self.lr,
        )
        return optimizer_generator, optimizer_discriminator

    @staticmethod
    def add_argparse_args(parser: ArgumentParser) -> ArgumentParser:
        model_parser = parser.add_argument_group("Model")
        return parser
