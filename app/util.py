import sys
import typer
from typing_extensions import Annotated
from pathlib import Path
from enum import Enum


from pytorchstudy.util.loguru_setup import logger, console_logger, file_logger


class TorchTask(str, Enum):
    si_snr = "si-snr"
    wav_conv = "wav-conv"


app = typer.Typer()


@app.command(help="pick wav files from dataset to the given folder")
def pick_wav(
    in_dir: Annotated[
        Path,
        typer.Option(
            # default=...,
            exists=True,
            file_okay=False,
            dir_okay=True,
            help="input dir of dataset",
        ),
    ],
    out_dir: Annotated[
        Path,
        typer.Option(
            # default=...,
            exists=True,
            file_okay=False,
            dir_okay=True,
            help="output dir of dataset",
        ),
    ],
    csv_file: Annotated[
        str,
        typer.Option(
            help="CSV file which contains the sample path",
        ),
    ],
    sample_id: Annotated[
        int,
        typer.Option(
            min=0,
            help="the sample id to pick",
        ),
    ],
):
    from pytorchstudy.util.dataset import pick_from_ds
    from returns.result import Failure, Success

    print(f"input dir is: {in_dir}")
    print(f"output dir is: {out_dir}")
    print(f"csv file is: {csv_file}")
    print(f"sample id: {sample_id}")

    if not (in_dir / csv_file).exists():
        print(f"{in_dir / csv_file} doesn't exist")
        raise typer.Exit(-1)

    ret = pick_from_ds(in_dir, out_dir, csv_file, sample_id)
    match ret:
        case Success(_):
            print("Successfully done")
        case Failure(Exception() as ex):
            print(f"get a failure:\n{ex}")
            raise typer.Exit(-1)


@app.command(help="list all cuda devices")
def list_cuda():
    import torch

    console_logger.debug("list cuda starts ...")
    device_count = torch.cuda.device_count()
    if device_count == 0:
        # console_logger.debug("No CUDA devices found.")
        return

    file_logger.debug(f"Number of CUDA devices available: {device_count}")
    for device_idx in range(device_count):
        device_name = torch.cuda.get_device_name(device_idx)
        console_logger.debug(f"Device {device_idx}: {device_name}")

    # Specify the device type
    torch.cuda.is_available()

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device


@app.command(help="Calculate SI-SNR")
def si_snr(
    pred_fn: Annotated[
        Path,
        typer.Option(
            exists=True,
            file_okay=True,
            dir_okay=False,
            help="Predicted soundtrack",
        ),
    ],
    target_fn: Annotated[
        Path,
        typer.Option(
            exists=True,
            file_okay=True,
            dir_okay=False,
            help="Groudtruth target soundtrack",
        ),
    ],
):
    """_summary_

    Args:
        pred_fn (Annotated[ Path, typer.Option, optional): _description_. Defaults to True, file_okay=True, dir_okay=False, help="Predicted soundtrack", ), ].
        target_fn (Annotated[ Path, typer.Option, optional): _description_. Defaults to True, file_okay=True, dir_okay=False, help="Groudtruth target soundtrack", ), ].

    https://torchmetrics.readthedocs.io/en/stable/audio/scale_invariant_signal_noise_ratio.html
    """
    import torch
    from torchmetrics import ScaleInvariantSignalNoiseRatio
    import librosa

    pred_y, pred_sr = librosa.load(str(pred_fn), sr=None, mono=True)
    target_y, target_sr = librosa.load(str(target_fn), sr=None, mono=True)

    if pred_sr != target_sr:
        print(f"signal wav sr [{pred_sr}] is different from noise wav sr [{target_sr}]")
        typer.Exit(-1)

    lp, lt = len(pred_y), len(target_y)
    if lp != lt:
        print(
            f"two wav files don't have the equal length (signal [{lp}], noise [{lt}], the shorter length will be used"
        )
        if lp > lt:
            print(f"target is the shorter one and will be used")
            pred_y = pred_y[:lt]
        else:
            print(f"predict is the shorter one and will be used")
            target_y = target_y[:lp]

    target = torch.tensor(target_y)
    preds = torch.tensor(pred_y)
    si_snr = ScaleInvariantSignalNoiseRatio()
    ret = si_snr(preds, target)
    print(f"SI-SNR: {ret}")


if __name__ == "__main__":
    print(f"python version is {sys.version_info}")
    if not (sys.version_info.major == 3 and sys.version_info.minor >= 9):
        sys.exit("this program needs python 3.10 and above to run")

    # https://towardsdatascience.com/a-simple-guide-to-command-line-arguments-with-argparse-6824c30ab1c3
    print(f"sys.path:\n{sys.path}")

    app()
