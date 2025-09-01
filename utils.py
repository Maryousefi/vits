import os
import glob
import sys
import argparse
import logging
import json
import subprocess
import numpy as np
from scipy.io.wavfile import read
import torch

MATPLOTLIB_FLAG = False

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging


def load_checkpoint(checkpoint_path, model, optimizer=None):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    iteration = checkpoint_dict["iteration"]
    learning_rate = checkpoint_dict["learning_rate"]
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint_dict["optimizer"])
    saved_state_dict = checkpoint_dict["model"]
    state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[k] = saved_state_dict.get(k, v)
        if k not in saved_state_dict:
            logger.info("%s missing in checkpoint", k)
    if hasattr(model, "module"):
        model.module.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(new_state_dict)
    logger.info("Loaded checkpoint %s (iteration %d)", checkpoint_path, iteration)
    return model, optimizer, learning_rate, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path):
    logger.info("Saving model checkpoint at iter %d to %s", iteration, checkpoint_path)
    state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    torch.save(
        {
            "model": state_dict,
            "iteration": iteration,
            "optimizer": optimizer.state_dict(),
            "learning_rate": learning_rate,
        },
        checkpoint_path,
    )


def summarize(writer, global_step, scalars={}, histograms={}, images={}, audios={}, audio_sampling_rate=22050):
    for k, v in scalars.items():
        writer.add_scalar(k, v, global_step)
    for k, v in histograms.items():
        writer.add_histogram(k, v, global_step)
    for k, v in images.items():
        writer.add_image(k, v, global_step, dataformats="HWC")
    for k, v in audios.items():
        writer.add_audio(k, v, global_step, audio_sampling_rate)


def latest_checkpoint_path(dir_path, regex="G_*.pth"):
    f_list = glob.glob(os.path.join(dir_path, regex))
    f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    return f_list[-1]


def plot_spectrogram_to_numpy(spectrogram):
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import matplotlib
        matplotlib.use("Agg")
        MATPLOTLIB_FLAG = True
        logging.getLogger("matplotlib").setLevel(logging.WARNING)
    import matplotlib.pylab as plt

    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


def plot_alignment_to_numpy(alignment, info=None):
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import matplotlib
        matplotlib.use("Agg")
        MATPLOTLIB_FLAG = True
        logging.getLogger("matplotlib").setLevel(logging.WARNING)
    import matplotlib.pylab as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(alignment.transpose(), aspect="auto", origin="lower", interpolation="none")
    fig.colorbar(im, ax=ax)
    xlabel = "Decoder timestep"
    if info is not None:
        xlabel += "\n\n" + info
    plt.xlabel(xlabel)
    plt.ylabel("Encoder timestep")
    plt.tight_layout()
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding="utf-8") as f:
        return [line.strip().split(split) for line in f]


def get_hparams(init=True):
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="./configs/base.json")
    parser.add_argument("-m", "--model", type=str, required=True)
    args = parser.parse_args()
    model_dir = os.path.join("./logs", args.model)
    os.makedirs(model_dir, exist_ok=True)
    config_save_path = os.path.join(model_dir, "config.json")
    if init:
        with open(args.config, "r", encoding="utf-8") as f:
            data = f.read()
        with open(config_save_path, "w", encoding="utf-8") as f:
            f.write(data)
    else:
        with open(config_save_path, "r", encoding="utf-8") as f:
            data = f.read()
    return HParams(**json.loads(data), model_dir=model_dir)


def get_hparams_from_dir(model_dir):
    config_save_path = os.path.join(model_dir, "config.json")
    with open(config_save_path, "r", encoding="utf-8") as f:
        data = f.read()
    return HParams(**json.loads(data), model_dir=model_dir)


def get_hparams_from_file(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        data = f.read()
    return HParams(**json.loads(data))


def check_git_hash(model_dir):
    source_dir = os.path.dirname(os.path.realpath(__file__))
    if not os.path.exists(os.path.join(source_dir, ".git")):
        logger.warning("%s not a git repo, ignoring hash check", source_dir)
        return
    cur_hash = subprocess.getoutput("git rev-parse HEAD")
    path = os.path.join(model_dir, "githash")
    if os.path.exists(path):
        saved_hash = open(path).read()
        if saved_hash != cur_hash:
            logger.warning("git hash mismatch: %s != %s", saved_hash[:8], cur_hash[:8])
    else:
        open(path, "w").write(cur_hash)


def get_logger(model_dir, filename="train.log"):
    logger = logging.getLogger(os.path.basename(model_dir))
    logger.setLevel(logging.DEBUG)
    os.makedirs(model_dir, exist_ok=True)
    handler = logging.FileHandler(os.path.join(model_dir, filename), encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s"))
    logger.addHandler(handler)
    return logger


class HParams:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            v = HParams(**v) if isinstance(v, dict) else v
            setattr(self, k, v)

    def __getitem__(self, key): return getattr(self, key)
    def __setitem__(self, key, value): return setattr(self, key, value)
    def __contains__(self, key): return key in self.__dict__
    def keys(self): return self.__dict__.keys()
    def items(self): return self.__dict__.items()
    def values(self): return self.__dict__.values()
    def __len__(self): return len(self.__dict__)
    def __repr__(self): return repr(self.__dict__)
