import os
from argparse import ArgumentParser

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch import optim

from shared.model.common import criterion
from shared.model.data.dataloader import get_data_loaders
from shared.model.data.dataset import ProductsDataset
from shared.model.mlp import get_mlp_model


class MlpModel(pl.LightningModule):
    def __init__(self, input_shape: int = 2329, output_shape: int = 22):
        super().__init__()
        self._mlp = get_mlp_model(input_shape, output_shape)

    def forward(self, value):
        return self._mlp(value)

    def training_step(self, batch, _):
        input, target = batch

        target = torch.argmax(target, axis=1)
        output, _ = self._mlp(input)

        train_loss = criterion(output, target)
        self.log("train_loss", train_loss)
        return train_loss

    def validation_step(self, batch, _):
        input, target = batch

        target = torch.argmax(target, axis=1)
        output, _ = self._mlp(input)

        validation_loss = criterion(output, target)
        self.log("validation_loss", validation_loss, on_epoch=True)
        return validation_loss

    def configure_optimizers(self):
        optimizers = optim.Adam(self._mlp.parameters(), lr=1e-4)
        return optimizers


def main(
    num_workers: int,
    batch_size: int,
    shared_path: str = "../../shared",
    complete_file_name: str = "amz_products_small.jsonl.gz",
):
    torch.manual_seed(1)
    model_train = MlpModel()
    products_dataset = ProductsDataset(shared_path + "/data", complete_file_name)
    train_loader, valid_loader = get_data_loaders(
        products_dataset, batch_size=batch_size, num_workers=num_workers
    )

    # If we had multiple GPUs, could gpus=2, accelerator='dp'.
    os.makedirs(shared_path + "/data/model_ckpt", exist_ok=True)
    early_stop_callback = EarlyStopping("validation_loss")
    checkpoint_callback = ModelCheckpoint(
        monitor="validation_loss",
        dirpath=shared_path + "/data/" + "model_ckpt/",
        filename="epoch={epoch}-validation_loss={validation_loss:.2f}",
        save_top_k=1,
    )

    # tb_logger = pl_loggers.TensorBoardLogger(save_dir=shared_path + '/' + "lightning_logs/")
    # comet_logger = pl_loggers.CometLogger(save_dir=shared_path + '/' + "lightning_logs/")

    trainer = pl.Trainer(
        max_epochs=5,
        callbacks=[early_stop_callback, checkpoint_callback],
    )
    trainer.fit(model_train, train_loader, valid_loader)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--workers", default=1)
    parser.add_argument("--batch_size", default=16)
    parser.add_argument("--shared_path", default="../../shared")
    parser.add_argument("--complete_file_name", default="amz_products_small.jsonl.gz")
    args = parser.parse_args()

    main(args.workers, args.batch_size, args.shared_path, args.complete_file_name)
