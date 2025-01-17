from typing import Callable, TypeVar, Optional
from dataclasses import dataclass

from omegaconf import OmegaConf, SCMode, MISSING

from .data.datamodule import ConfData
from .data.preprocess import ConfMelspectrogram
from .train import ConfTrain
from .model import ConfRNNMS


CONF_DEFAULT_STR = """
seed: 1234
path_extend_conf: null
target_sr: 16000
bits_mulaw: 10
dim_mel: 80
win_length: 800
stride_stft: 200
melspec:
    sr: "${target_sr}"
    n_fft: "${win_length}"
    hop_length: "${stride_stft}"
    preemph: 0.97
    top_db: 80.0
    ref_db: 20.0
    n_mels: "${dim_mel}"
    fmin: 50
    fmax: null
model:
    sampling_rate: "${target_sr}"
    vocoder:
        dim_i_feature: "${dim_mel}"
        dim_voc_latent: 256
        bits_mu_law: "${bits_mulaw}"
        upsampling_t: "${stride_stft}"
        prenet:
            num_layers: 2
            bidirectional: True
        wave_ar:
            recurrent: GRU
            size_i_embed_ar: 256
            size_h_rnn: 896
            size_h_fc: 1024
    optim:
        learning_rate: 4.0e-4
        sched_decay_rate: 0.5
        sched_decay_step: 25000
    wav2mel: "${melspec}"
data:
    data_name: LJ
    adress_data_root: null
    loader:
        batch_size: 32
        num_workers: null
        pin_memory: null
    dataset:
        clip_length_mel: 24
        mel_stft_stride: "${stride_stft}"
        preprocess:
            target_sr: "${target_sr}"
            win_length: "${win_length}"
            bits_mulaw: "${bits_mulaw}"
            melspec: "${melspec}"
    corpus:
        download: False
train:
    max_epochs: 2000
    val_interval_epoch: 20
    profiler: null
    ckpt_log:
        dir_root: logs
        name_exp: default
        name_version: version_-1
"""


@dataclass
class ConfGlobal:
    """Configuration of everything.
    Args:
        seed: PyTorch-Lightning's seed for every random system
        path_extend_conf: Path of configuration yaml which extends default config
        target_sr: Desired sampling rate of waveform
        bits_mulaw: Bit depth of μ-law compressed waveform
        dim_mel: Dimension of mel-spectrogram
        stride_stft: STFT stride
        win_length: STFT window length
    """
    seed: int = MISSING
    path_extend_conf: Optional[str] = MISSING
    target_sr: int = MISSING
    bits_mulaw: int = MISSING
    dim_mel: int = MISSING
    stride_stft: int = MISSING
    win_length: int = MISSING
    melspec: ConfMelspectrogram = ConfMelspectrogram()
    model: ConfRNNMS = ConfRNNMS()
    data: ConfData = ConfData()
    train: ConfTrain = ConfTrain()


T = TypeVar('T')
def gen_load_conf() -> Callable[[], T]:
    """Generate 'Load configuration type-safely' function.

    Priority: CLI args > CLI-specified config yaml > Default
    """

    def generated_load_conf() -> T:
        default = OmegaConf.create(CONF_DEFAULT_STR)
        cli = OmegaConf.from_cli()
        extends_path = cli.get("path_extend_conf", None)
        if extends_path:
            extends = OmegaConf.load(extends_path)
            conf_final = OmegaConf.merge(default, extends, cli)
        else:
            conf_final = OmegaConf.merge(default, cli)
        OmegaConf.resolve(conf_final)
        conf_structured = OmegaConf.merge(
            OmegaConf.structured(ConfGlobal),
            conf_final
        )

        # Design Note -- OmegaConf instance v.s. DataClass instance --
        #   OmegaConf instance has runtime overhead in exchange for type safety.
        #   Configuration is constructed/finalized in early stage,
        #   so config is eternally valid after validation in last step of early stage.
        #   As a result, we can safely convert OmegaConf to DataClass after final validation.
        #   This prevent (unnecessary) runtime overhead in later stage.
        #
        #   One demerit: No "freeze" mechanism in instantiated dataclass.
        #   If OmegaConf, we have `OmegaConf.set_readonly(conf_final, True)`

        # [todo]: Return both dataclass and OmegaConf because OmegaConf has export-related utils.

        # `.to_container()` with `SCMode.INSTANTIATE` resolve interpolations and check MISSING.
        # It is equal to whole validation.
        return OmegaConf.to_container(conf_structured, structured_config_mode=SCMode.INSTANTIATE)

    return generated_load_conf

load_conf = gen_load_conf()
"""Load configuration type-safely.
"""
