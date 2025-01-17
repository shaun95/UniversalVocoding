<div align="center">

# RNN_MS-PyTorch <!-- omit in toc -->
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)][notebook]
[![Paper](http://img.shields.io/badge/paper-arxiv.1811.06292-B31B1B.svg)][paper]  

</div>

Neural vocoder **"RNN_MS"** with PyTorch.

<!-- generated by [Markdown All in One](https://marketplace.visualstudio.com/items?itemName=yzhang.markdown-all-in-one) -->
- [Demo](#demo)
- [Quick training](#quick-training)
- [How to Use](#how-to-use)
- [System Details](#system-details)
- [Results](#results)
- [References](#references)

## Demo
<!-- [Audio sample page](https://tarepan.github.io/UniversalVocoding).   -->
ToDO: Link super great impressive high-quatity audio demo.  

## Quick Training
Jump to ☞ [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)][notebook], then Run. That's all!  

## How to Use
### 1. Install <!-- omit in toc -->

```bash
# pip install "torch==1.10.0" -q      # Based on your environment (validated with v1.10)
# pip install "torchaudio==0.10.0" -q # Based on your environment
pip install git+https://github.com/tarepan/UniversalVocoding
```

### 2. Data & Preprocessing <!-- omit in toc -->
"Batteries Included".  
RNNMS transparently download corpus and preprocess it for you 😉  

### 3. Train <!-- omit in toc -->
```bash
python -m rnnms.main_train
```

For arguments, check [./rnnms/config.py](https://github.com/tarepan/UniversalVocoding/blob/main/rnnms/config.py)  

#### Advanced: Other datasets <!-- omit in toc -->
You can switch dataset with arguments.  
All [`speechcorpusy`](https://github.com/tarepan/speechcorpusy)'s preset corpuses are supported.  

```bash
# LJSpeech corpus
python -m rnnms.main_train data.data_name=LJ
```

#### Advanced: Custom dataset <!-- omit in toc -->
Copy [`rnnms.main_train`] and replace DataModule.  

```python
    # datamodule = LJSpeechDataModule(batch_size, ...)
    datamodule = YourSuperCoolDataModule(batch_size, ...)
    # That's all!
```

[`rnnms.main_train`]:https://github.com/tarepan/UniversalVocoding/blob/main/rnnms/main_train.py

### 4. Inference <!-- omit in toc -->
```bash
python -m rnnms.main_inference model_ckpt_path=XXX i_wav_path=YYY o_wav_path=ZZZ
```

## System Details
### Model <!-- omit in toc -->
- PreNet: 2-layer bidi-GRU
- Upsampler: x200 time-directional latent upsampling with interpolation
- Decoder: Latent-conditional, embedded-auto-regressive generative RNN with 10-bit μ-law encoding

### Differences from the Paper <!-- omit in toc -->

| property      |  paper           | this repo       |
|:--------------|:-----------------|:----------------|
| sampling rate | 24 kHz           | 16 kHz          |
| AR input      | one-hot          | embedding       |
| Dataset       | internal? 74 spk | JVS, 100 spk    |
| Presicion     |   -              | 32/16 Mixed     |

## Results
### Output Sample <!-- omit in toc -->
[Demo](#demo)

### Performance <!-- omit in toc -->
1.1 [iter/sec] @ NVIDIA T4 on Google Colaboratory (AMP+, num_workers=8)  
(1.1 [iter/sec] with [bshall/UniversalVocoding], same setup)  

It takes about 2days for full training.  

## References
### Original paper <!-- omit in toc -->
[![Paper](http://img.shields.io/badge/paper-arxiv.1811.06292-B31B1B.svg)][paper]  
<!-- https://arxiv2bibtex.org/?q=1811.06292&format=bibtex -->
```
@misc{1811.06292,
Author = {Jaime Lorenzo-Trueba and Thomas Drugman and Javier Latorre and Thomas Merritt and Bartosz Putrycz and Roberto Barra-Chicote and Alexis Moinet and Vatsal Aggarwal},
Title = {Towards achieving robust universal neural vocoding},
Year = {2018},
Eprint = {arXiv:1811.06292},
}
```

### Acknowlegements <!-- omit in toc -->
- [bshall/UniversalVocoding]: Model and hyperparams are totally based on this repository. All codes are re-written.


[paper]:https://arxiv.org/abs/1811.06292
[notebook]:https://colab.research.google.com/github/tarepan/UniversalVocoding/blob/main/rnnms.ipynb
[bshall/UniversalVocoding]:https://github.com/bshall/UniversalVocoding
