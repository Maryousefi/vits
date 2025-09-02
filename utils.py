import os
import glob
import sys
import argparse
import logging
import json
import subprocess
import re
import numpy as np
from scipy.io.wavfile import read
import torch

MATPLOTLIB_FLAG = False

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging


def load_checkpoint(checkpoint_path, model, optimizer=None):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    iteration = checkpoint_dict.get("iteration", 0)
    learning_rate = checkpoint_dict.get("learning_rate", None)
    if optimizer is not None and "optimizer" in checkpoint_dict:
        optimizer.load_state_dict(checkpoint_dict["optimizer"])
    saved_state_dict = checkpoint_dict.get("model", {})
    state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():
        if k in saved_state_dict:
            new_state_dict[k] = saved_state_dict[k]
        else:
            new_state_dict[k] = v
            logger.info("%s is not in the checkpoint", k)
    if hasattr(model, "module"):
        model.module.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(new_state_dict)
    logger.info("Loaded checkpoint '%s' (iteration %s)", checkpoint_path, iteration)
    return model, optimizer, learning_rate, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path):
    logger.info("Saving model and optimizer state at iteration %s to %s", iteration, checkpoint_path)
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


def _extract_number_from_filename(path):
    """Return last integer found in filename or None"""
    base = os.path.basename(path)
    nums = re.findall(r"\d+", base)
    if not nums:
        return None
    return int(nums[-1])


def latest_checkpoint_path(dir_path, regex="G_*.pth"):
    """
    Return the most recent checkpoint according to numeric suffix in filename.
    If no numeric suffixes found, use lexicographical order and return last.
    Raises FileNotFoundError when no files match.
    """
    f_list = glob.glob(os.path.join(dir_path, regex))
    if len(f_list) == 0:
        raise FileNotFoundError(f"No checkpoints found with pattern {os.path.join(dir_path, regex)}")
    # Sort by numeric suffix if present, otherwise lexicographically
    def sort_key(f):
        n = _extract_number_from_filename(f)
        if n is None:
            # push non-numeric names to the end (so G_1000.pth will be before G_final.pth)
            return (1, f)
        return (0, n)
    f_list.sort(key=sort_key)
    return f_list[-1]


def plot_spectrogram_to_numpy(spectrogram):
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import matplotlib

        matplotlib.use("Agg")
        MATPLOTLIB_FLAG = True
        mpl_logger = logging.getLogger("matplotlib")
        mpl_logger.setLevel(logging.WARNING)
    import matplotlib.pylab as plt
    import numpy as np

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
        mpl_logger = logging.getLogger("matplotlib")
        mpl_logger.setLevel(logging.WARNING)
    import matplotlib.pylab as plt
    import numpy as np

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
        filepaths_and_text = [line.strip().split(split) for line in f if line.strip()]
    return filepaths_and_text


def get_hparams(init=True):
    """
    Load JSON config and return HParams object.
    Also pre-load symbols if text_cleaners includes Persian.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="./configs/base.json", help="JSON file for configuration")
    parser.add_argument("-m", "--model", type=str, required=True, help="Model name")
    args = parser.parse_args()

    model_dir = os.path.join("./logs", args.model)
    os.makedirs(model_dir, exist_ok=True)

    config_path = args.config
    config_save_path = os.path.join(model_dir, "config.json")
    if init:
        with open(config_path, "r", encoding="utf-8") as f:
            data = f.read()
        with open(config_save_path, "w", encoding="utf-8") as f:
            f.write(data)
    else:
        with open(config_save_path, "r", encoding="utf-8") as f:
            data = f.read()

    config = json.loads(data)
    hparams = HParams(**config)
    hparams.model_dir = model_dir

    # Auto-load symbols if Persian cleaner requested
    try:
        if hasattr(hparams, "data") and hasattr(hparams.data, "text_cleaners"):
            cleaners = hparams.data.text_cleaners
            if cleaners:
                # Delay import, text.get_symbols is robust and returns list of symbols
                from text import get_symbols

                symbols = get_symbols(cleaners)
                # Store symbols and count for downstream use
                setattr(hparams.data, "symbols", symbols)
                setattr(hparams.data, "n_symbols", len(symbols))
                logger.info("Auto-loaded symbols based on cleaners=%s (n_symbols=%d)", cleaners, len(symbols))
    except Exception as e:
        logger.warning("Auto-load of symbols failed: %s", e)

    return hparams


def get_hparams_from_dir(model_dir):
    config_save_path = os.path.join(model_dir, "config.json")
    with open(config_save_path, "r", encoding="utf-8") as f:
        data = f.read()
    config = json.loads(data)
    hparams = HParams(**config)
    hparams.model_dir = model_dir
    return hparams


def get_hparams_from_file(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        data = f.read()
    config = json.loads(data)
    hparams = HParams(**config)
    return hparams


def check_git_hash(model_dir):
    source_dir = os.path.dirname(os.path.realpath(__file__))
    if not os.path.exists(os.path.join(source_dir, ".git")):
        logger.warn("%s is not a git repository, therefore hash value comparison will be ignored.", source_dir)
        return

    cur_hash = subprocess.getoutput("git rev-parse HEAD")

    path = os.path.join(model_dir, "githash")
    if os.path.exists(path):
        saved_hash = open(path).read()
        if saved_hash != cur_hash:
            logger.warn("git hash values are different. %s (saved) != %s (current)", saved_hash[:8], cur_hash[:8])
    else:
        open(path, "w").write(cur_hash)


def get_logger(model_dir, filename="train.log"):
    global logger
    # Avoid adding multiple handlers if called multiple times
    logger = logging.getLogger(os.path.basename(model_dir))
    logger.setLevel(logging.DEBUG)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    # If handler already exists, return logger
    if logger.handlers:
        return logger
    h = logging.FileHandler(os.path.join(model_dir, filename), encoding="utf-8")
    h.setLevel(logging.DEBUG)
    h.setFormatter(logging.Formatter("%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s"))
    logger.addHandler(h)
    return logger


class HParams:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, dict):
                v = HParams(**v)
            setattr(self, k, v)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __repr__(self):
        return repr(self.__dict__)
