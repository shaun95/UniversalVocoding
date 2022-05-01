"""Run RNNMS training"""


import pytorch_lightning as pl
import torchaudio

from rnnms.model import RNNMS
from rnnms.data.datamodule import generate_datamodule
from rnnms.train import train
from rnnms.config import load_conf


def main_train():
    """Train rnnms with cli arguments and the default dataset.
    """

    # Load default/extend/CLI configs.
    conf = load_conf()

    # Setup
    pl.seed_everything(conf.seed)
    torchaudio.set_audio_backend("sox_io")
    model = RNNMS(conf.model)
    datamodule = generate_datamodule(conf.data)

    # Train
    train(model, conf.train, datamodule)


if __name__ == "__main__":  # pragma: no cover
    main_train()
