import os
import numpy as np
import matplotlib.pyplot as plt
from ssqueezepy import ssq_cwt, ssq_stft
from ssqueezepy.experimental import scale_to_freq


os.environ["SSQ_GPU"] = "1"
N = 2048


def viz(x, Tx, Wx):
    plt.imshow(np.abs(Wx), aspect="auto", cmap="turbo")
    plt.show()
    plt.imshow(np.abs(Tx), aspect="auto", vmin=0, vmax=0.2, cmap="turbo")
    plt.show()


def gen_signals():
    # Define signal ####################################
    t = np.linspace(0, 10, N, endpoint=False)
    xo = np.cos(2 * np.pi * 2 * (np.exp(t / 2.2) - 1))
    xo += xo[::-1]  # add self reflected
    x = xo + np.sqrt(2) * np.random.randn(N)  # add noise

    return xo, x


def plot_ssq_cwt(xo, x):
    # CWT + SSQ CWT ####################################
    Twxo, Wxo, *_ = ssq_cwt(xo)
    viz(xo, Twxo, Wxo)

    Twx, Wx, *_ = ssq_cwt(x)
    viz(x, Twx, Wx)


def plot_ssq_stft(xo, x):
    # STFT + SSQ STFT ##################################
    Tsxo, Sxo, *_ = ssq_stft(xo)
    viz(xo, np.flipud(Tsxo), np.flipud(Sxo))

    Tsx, Sx, *_ = ssq_stft(x)
    viz(x, np.flipud(Tsx), np.flipud(Sx))


def plot_with_unit(x):
    # With units #######################################
    from ssqueezepy import Wavelet, cwt, stft, imshow

    fs = 400
    t = np.linspace(0, N / fs, N)
    wavelet = Wavelet()
    Wx, scales = cwt(x, wavelet)
    Sx = stft(x)[::-1]

    freqs_cwt = scale_to_freq(scales, wavelet, len(x), fs=fs)
    freqs_stft = np.linspace(1, 0, len(Sx)) * fs / 2

    ikw = dict(abs=1, xticks=t, xlabel="Time [sec]", ylabel="Frequency [Hz]")
    imshow(Wx, **ikw, yticks=freqs_cwt)
    imshow(Sx, **ikw, yticks=freqs_stft)


def replace_zeroes(data):
    min_nonzero = np.min(np.abs(data[np.nonzero(data)]))
    # logger.debug(f"min_nonzero: {min_nonzero}")
    data[data == 0] = min_nonzero
    # data[data == 0] = 0.00001
    return data


def plot_gh(x, fs):
    # With units #######################################
    from ssqueezepy import Wavelet, cwt, imshow

    N = len(x)
    t = np.linspace(0, N / fs, N)
    wavelet = Wavelet(wavelet="morlet")
    Wx, scales = cwt(x, wavelet, fs=fs, astensor=False, nv=12)
    # Wx, scales = Wx0.cpu(), scales0.cpu()

    # freqs_cwt = scale_to_freq(scales, wavelet, len(x), fs=fs)

    # ikw = dict(abs=1, xticks=t, xlabel="Time [sec]", ylabel="Frequency [Hz]")
    # imshow(Wx, **ikw, yticks=freqs_cwt, cmap="magma")
    # imshow(20 * np.log10(np.abs(Wx)), **ikw, yticks=freqs_cwt, cmap="magma")

    dim_w, dim_h = 6, 3  # convert_dim_inch(t, w, h, dpi)
    cmap, threshold = "magma", -60
    dpi = 256

    Wx = replace_zeroes(Wx)

    fig, ax = plt.subplots(1, 1, figsize=(dim_w, dim_h), dpi=dpi)
    ax.set_xlabel("Time")
    ax.set(title="Scalogram")
    img = ax.imshow(
        20 * np.log10(np.abs(Wx)),
        cmap=cmap,
        aspect="auto",
        norm=None,
        vmax=0,
        vmin=threshold,
        extent=[0.0, len(x) / float(fs), Wx.shape[0], 0],
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    fig.savefig("test.png")
