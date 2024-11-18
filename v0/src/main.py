# define launching pytorch lighting here


from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import wandb
from pytorch_lightning import seed_everything
from model import mmContrastiveModel
from dataset import medicalDataModule
import pandas as pd
import os
import torch
import getpass
from absl import flags
from absl import app

_CLIP_MODEL = flags.DEFINE_string(
    "clip_model", "openai/clip-vit-base-patch32", "CLIP model to use"
)


def main(argv):
    del argv  # unused
    clip_model = _CLIP_MODEL.value
    seed_everything(42)
    # init model
    model = mmContrastiveModel(model_path=clip_model)
    clip_name = clip_model.split("/")[-1]
    # init dataset
    # debug only loads the first 25 examples from each dataset file

    user = getpass.getuser()
    local_machine = False
    dataset = medicalDataModule(
        model_path=clip_model,
        tensor_dir=clip_name + "-MedicalTensors-Final/",
        batch_size=64,
        debug=False,
        force_remake=False,
        local_machine=local_machine,
    )
    torch.set_float32_matmul_precision("medium")

    # init trainer
    trainer = Trainer(
        accelerator="gpu",
        max_epochs=30,
        max_steps=1607 * 30,
        logger=wandb.WandbLogger(
            project="MedGeese",
            save_dir=user + "-logs",
            name=f"Full Dataset With Transforms - Header Only - {clip_model}",
            offline=False,
        ),  # need offline for CC, can do online on lg-2
        callbacks=[
            ModelCheckpoint(monitor="val/loss", mode="min"),
            LearningRateMonitor(logging_interval="step"),
        ],
        log_every_n_steps=50,
        gradient_clip_val=5.0,
        precision="16-mixed",
    )

    # train model
    trainer.fit(model, datamodule=dataset)

    if local_machine:
        print("attempting to clean up")
        os.system("rm -r /tmp/*-MedicalTensors-*")


if __name__ == "__main__":
    app.run(main)
