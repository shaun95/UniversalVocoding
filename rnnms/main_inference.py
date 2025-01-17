"""Run RNNMS inference"""


import argparse

from torch import inference_mode
import torchaudio
import librosa
import soundfile as sf

from rnnms.model import RNNMS


if __name__ == "__main__":  # pragma: no cover

    parser = argparse.ArgumentParser(description='Run RNNMS inference')
    parser.add_argument("-m", "--model_ckpt_path", required=True)
    parser.add_argument("-i", "--i_wav_path",      required=True)
    parser.add_argument("-o", "--o_wav_path", default="reconstructed.wav")
    args = parser.parse_args()

    torchaudio.set_audio_backend("sox_io")
    model = RNNMS.load_from_checkpoint(checkpoint_path=args.model_ckpt_path)
    wave, orig_sr = librosa.load(args.i_wav_path)

    with inference_mode():
        mel = model.wav2mel(wave, orig_sr)
        o_wave, o_sr = model.predict(mel)
    # Tensor[1, T] => Tensor[T,] => ndarray[T,]
    o_wave = o_wave[0].to('cpu').detach().numpy()

    sf.write(args.o_wav_path, o_wave, o_sr)
