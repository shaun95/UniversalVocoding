"""RNNMS PyTorch-Lightnig model"""


from typing import Tuple
from dataclasses import dataclass

import numpy as np
from torch import Tensor, FloatTensor, reshape
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import pytorch_lightning as pl
from omegaconf import MISSING
import librosa

from .networks.vocoder import RNNMSVocoder, ConfRNNMSVocoder
from .data.preprocess import melspectrogram, ConfMelspectrogram


@dataclass
class ConfOptim:
    """Configuration of optimizer.
    Args:
        learning_rate: Optimizer learning rate
        sched_decay_rate: LR shaduler decay rate
        sched_decay_step: LR shaduler decay step
    """
    learning_rate: float = MISSING
    sched_decay_rate: float = MISSING
    sched_decay_step: int = MISSING

@dataclass
class ConfRNNMS:
    """Configuration of RNN_MS.
    """
    sampling_rate: int = MISSING
    vocoder: ConfRNNMSVocoder = ConfRNNMSVocoder()
    optim: ConfOptim = ConfOptim()
    wav2mel: ConfMelspectrogram = ConfMelspectrogram()

class RNNMS(pl.LightningModule):
    """RNN_MS, universal neural vocoder.
    """

    def __init__(self, conf: ConfRNNMS):
        super().__init__()
        self.save_hyperparameters()
        self.conf = conf
        self.rnnms = RNNMSVocoder(conf.vocoder)

    def forward(self, mels: Tensor):
        """Generate a waveform from a log-mel spectrogram.

        Args:
            mels::Tensor[Batch==1, TimeMel, Freq] - Input log-mel spectrogram
        Returns:
            Tensor[Batch==1, TimeWave] - PCM waveform
        """
        return self.rnnms.generate(mels)

    def training_step(self, batch: Tuple[Tensor, Tensor]):
        """Supervised learning.
        """

        wave_mu_law, spec_mel = batch

        bits_energy_sereis = self.rnnms(wave_mu_law[:, :-1], spec_mel)
        loss = F.cross_entropy(bits_energy_sereis.transpose(1, 2), wave_mu_law[:, 1:])

        self.log('loss', loss)
        return {"loss": loss}

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int):
        """full length needed & padding is not good (long white seems to be not good for RNN)
        => cannot batch (batch=1)
        """

        _, mels = batch
        wave = self.rnnms.generate(mels)

        # [PyTorch](https://pytorch.org/docs/stable/tensorboard.html#torch.
        #     utils.tensorboard.writer.SummaryWriter.add_audio)
        # add_audio(tag: str, snd_tensor: Tensor(1, L),
        #     global_step: Optional[int] = None, sample_rate: int = 44100)
        self.logger.experiment.add_audio(
            f"audio_{batch_idx}",
            wave,
            global_step=self.global_step,
            sample_rate=self.conf.sampling_rate,
        )

        return {
            "val_loss": 0,
        }

    def configure_optimizers(self):
        """Set up a optimizer
        """
        conf = self.conf.optim

        optim = Adam(self.rnnms.parameters(), lr=conf.learning_rate)
        sched = {
            "scheduler": StepLR(optim, conf.sched_decay_step, conf.sched_decay_rate),
            "interval": "step",
        }

        return {
            "optimizer": optim,
            "lr_scheduler": sched,
        }

    def predict(self, mels: Tensor) -> Tuple[Tensor, int]:
        """Generate a waveform from a log-mel spectrogram.

        Args:
            mels::Tensor[Batch==1, TimeMel, Freq] - Input log-mel spectrogram
        Returns:
            (Tensor[Batch==1, TimeWave], sr) - PCM waveform and its sampling rate
        """
        return self(mels), self.conf.sampling_rate

    def wav2mel(self, wave: np.ndarray, orig_sr: int) -> Tensor:
        """Convert a numpy PCM waveform into a PyTorch batched log-mel spectrogram

        Args:
            wave::(TimeWave,) - Target waveform
            orig_sr - Sampling rate of the original wave
        Returns::(Batch==1, TimeMel, Freq) - Batched log-mel spectrogram of resampled waveform
        """
        wave_resampled = librosa.resample(wave, orig_sr=orig_sr, target_sr=self.conf.sampling_rate)
        mel = FloatTensor(melspectrogram(wave_resampled, self.conf.wav2mel).T)
        mels = reshape(mel, (1, mel.shape[0], -1))
        return mels

    def sample_wave(self) -> Tuple[np.ndarray, int]:
        """Sample speech"""
        wave, sr = librosa.load(librosa.example("libri2"), sr=self.conf.sampling_rate)
        return wave, sr
