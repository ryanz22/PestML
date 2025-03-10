import os
import sys
import pathlib
import numpy as np
import pprint
from enum import Enum
from typing_extensions import Annotated
import typer
import logging
import functional as pyf
import shutil
from functools import partial

from pytorchstudy.ssqueezepy_trial import (
    gen_signals,
    plot_ssq_cwt,
    plot_ssq_stft,
    plot_with_unit,
    plot_gh,
)


app = typer.Typer()


class DemoTask(str, Enum):
    signal = "signal"
    ssq_cwt = "ssq_cwt"
    ssq_stft = "ssq_stft"
    cwt = "cwt"


@app.command(help="ssqueezepy demo")
def demo(
    task: Annotated[
        DemoTask,
        typer.Option(
            help="demo task to run",
        ),
    ],
):
    import matplotlib.pyplot as plt

    xo, x = gen_signals()
    if task == "signal":
        plt.plot(xo)
        plt.show()
        plt.plot(x)
        plt.show()
    elif task == "ssq_cwt":
        plot_ssq_cwt(xo, x)
    elif task == "ssq_stft":
        plot_ssq_stft(xo, x)
    elif task == "cwt":
        plot_with_unit(x)
    else:
        print(f"Unknown task: {task}")


@app.command(help="grasshopper ssqueezepy plot cwt")
def gh_plot(
    in_fn: Annotated[
        pathlib.Path,
        typer.Option(
            # default=...,
            exists=True,
            file_okay=True,
            dir_okay=False,
            help="grasshopper wav to plot",
        ),
    ],
):
    import librosa

    d1, sr1 = librosa.load(in_fn, sr=None, mono=True)
    plot_gh(d1, sr1)


@app.command(help="speechbrain DropFreq test")
def drop_freq(
    in_fn: Annotated[
        pathlib.Path,
        typer.Option(
            exists=True,
            file_okay=True,
            dir_okay=False,
            help="Predicted soundtrack",
        ),
    ],
):
    from speechbrain.dataio.dataio import read_audio
    from speechbrain.processing.speech_augmentation import DropFreq

    from pytorchstudy.util.plot import plot_fft_2

    SR = 44100

    # 44100 * 0.1 = 4410
    dropper = DropFreq(
        drop_prob=1, drop_freq_low=0.08, drop_count_low=1, drop_count_high=2000
    )
    signal = read_audio(str(in_fn))
    print(f"shape of original signal: {signal.shape}")
    dropped_signal = dropper(signal.unsqueeze(0))
    print(f"shape of dropped signal: {dropped_signal.shape}")
    plot_fft_2(
        (signal.numpy(), dropped_signal[-1].numpy()),
        sr=SR,
        out_fn="dropped_signal.png",
    )


@app.command(help="wave to conv matrix")
def wav_conv(
    in_fn: Annotated[
        pathlib.Path,
        typer.Option(
            exists=True,
            file_okay=True,
            dir_okay=False,
            help="Predicted soundtrack",
        ),
    ],
):
    """_summary_

    Args:
        in_fn (Annotated[ Path, typer.Option, optional): _description_. Defaults to True, file_okay=True, dir_okay=False, help="Predicted soundtrack", ), ].

    """
    import torch
    import librosa
    from speechbrain.processing.signal_processing import convolve1d

    y, sr = librosa.load(str(in_fn), sr=None, mono=True)

    signal = torch.tensor(y)
    print(f"signal shape after load wav: {signal.shape}")
    signal = signal.unsqueeze(0).unsqueeze(2)
    print(f"signal shape after unsqueeze: {signal.shape}")
    kernel = torch.rand(1, 10, 1)
    signal = convolve1d(signal, kernel, padding=(9, 0))
    print(f"signal shape after convolve: {signal.shape}")


@app.command(help="torch conv1d")
def torch_conv1d(
    in_chan: int = 1, out_chan: int = 100, kernel_size: int = 4, padding: int = 0
):
    from pytorchstudy.torch.conv import conv1d

    conv1d(in_chan=in_chan, out_chan=out_chan, kernel_size=kernel_size, padding=padding)


@app.command(help="torch conv transpose 1d")
def torch_deconv1d(
    in_chan: int = 1,
    out_chan: int = 100,
    kernel_size: int = 4,
    stride: int = 2,
    padding: int = 0,
    bias: bool = False,
):
    import torch
    from pytorchstudy.torch.conv import deconv1d

    y = torch.ones(1, 1, 2)
    print(f"y:\n{y}")

    w = torch.ones(1, 1, kernel_size)
    print(f"w:\n{w}")

    ret = deconv1d(
        y,
        w,
        in_chan=in_chan,
        out_chan=out_chan,
        ks=kernel_size,
        stride=stride,
        padding=padding,
        bias=bias,
    )
    print(ret)


if __name__ == "__main__":
    print(f"python version is {sys.version_info}")
    if not (sys.version_info.major == 3 and sys.version_info.minor >= 9):
        sys.exit("this program needs python 3.10 and above to run")

    # https://towardsdatascience.com/a-simple-guide-to-command-line-arguments-with-argparse-6824c30ab1c3
    print(f"sys.path:\n{sys.path}")

    # l_fmt = '[%(levelname)s] %(asctime)s - %(message)s'
    # logging.basicConfig(level=logging.ERROR, format=l_fmt)

    l_fmt = "[%(name)s %(levelname)s] %(asctime)s - %(message)s"
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(l_fmt))
    logger = logging.getLogger("dataset_tool")
    logger.addHandler(ch)
    logger.setLevel(logging.ERROR)

    app()
