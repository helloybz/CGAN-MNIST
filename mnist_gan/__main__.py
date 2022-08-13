from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers

from mnist_gan.dataset import MNISTDataModule
from mnist_gan.modules import GAN


def train(args):
    pl.seed_everything(42)

    dm = MNISTDataModule(**vars(args))
    model = GAN(**vars(args))

    logger = pl_loggers.TensorBoardLogger(
        save_dir=args.default_root_dir,
        name='mnist_gan',
    )
    checkpoint_manager = ModelCheckpoint(
        monitor='Loss/Generator/Valid',
        save_top_k=3,
    )
    trainer = pl.Trainer.from_argparse_args(
        args=args,
        logger=logger,
        callbacks=[checkpoint_manager],
    )
    trainer.fit(model, dm)
    trainer.test(model, dm)

def inference(args):
    model = GAN.load_from_checkpoint(args.checkpoint_path)
    generated_suite = model()
    images = MNISTDataModule.to_image(generated_suite)
    breakpoint()
    pass

def main():
    parser = ArgumentParser(prog='mnist_gan')
    subparsers = parser.add_subparsers()

    train_parser = subparsers.add_parser('train')
    train_parser.set_defaults(func=train)
    train_parser = pl.Trainer.add_argparse_args(train_parser)
    train_parser = MNISTDataModule.add_argparse_args(train_parser)
    train_parser = GAN.add_argparse_args(train_parser)

    inference_parser = subparsers.add_parser('inference')
    inference_parser.add_argument('--checkpoint_path')
    inference_parser.add_argument('--output_path', default='.')
    inference_parser.set_defaults(func=inference)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
