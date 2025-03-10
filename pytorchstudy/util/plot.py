import matplotlib.pyplot as plt
import numpy as np


def plot_fft(
    d,
    sr: int,
    out_fn: str,
    dim=("inch", 10, 4),
    show_scale: bool = False,
    dpi: int = 256,
):
    xf, yf = fft_process(d, sr)

    t, w, h = dim
    dim_w, dim_h = convert_dim_inch(t, w, h, dpi)

    fig, ax = plt.subplots(1, 1, figsize=(dim_w, dim_h))
    ax.plot(xf, np.abs(yf))

    if show_scale:
        ax.set(title="FFT")
        ax.set_xlabel("Frequency")
        ax.set_ylabel("Magnitude")
    else:
        ax.set_axis_off()

    fig.savefig(out_fn)


def plot_fft_2(
    dt: tuple,
    sr: int,
    out_fn: str,
    dim=("inch", 10, 10),
    show_scale: bool = True,
    dpi: int = 256,
):
    d1, d2 = dt
    xf1, yf1 = fft_process(d1, sr)
    xf2, yf2 = fft_process(d2, sr)

    t, w, h = dim
    dim_w, dim_h = convert_dim_inch(t, w, h, dpi)

    fig, ax = plt.subplots(2, 1, figsize=(dim_w, dim_h))
    ax[0].plot(xf1, np.abs(yf1))
    ax[1].plot(xf2, np.abs(yf2))

    if show_scale:
        ax[0].set(title="FFT")
        ax[0].set_xlabel("Frequency")
        ax[0].set_ylabel("Magnitude")
        ax[1].set(title="FFT")
        ax[1].set_xlabel("Frequency")
        ax[1].set_ylabel("Magnitude")
    else:
        ax[0].set_axis_off()
        ax[1].set_axis_off()

    fig.savefig(out_fn)


def fft_process(d2, sr):
    from scipy.fft import rfft, rfftfreq

    N = len(d2)
    # print(f"N: {N}")
    yf = rfft(d2)
    xf = rfftfreq(N, 1 / sr)
    # yf = np.fft.rfft(d2)
    # xf = np.fft.rfftfreq(N, 1 / sr)

    return xf, yf


def convert_dim_inch(t: str, w: float, h: float, dpi: int) -> tuple[float, float]:
    ow = w
    oh = h

    match t:
        case "cm":
            cm = 1 / 2.54
            ow = w * cm
            oh = h * cm
        case "px":
            print(f"calc px, dpi: {dpi}")
            px = 1 / dpi
            ow = w * px
            oh = h * px

    return ow, oh
