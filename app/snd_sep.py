import os
import sys
import pathlib
import numpy as np
import pprint
from typing import Tuple, List, Optional
import logging
import logging.config
import functional as pyf
import shutil
from enum import Enum
from typing_extensions import Annotated
import typer
from functools import partial
import yaml

from returns.result import Result, safe, Success, Failure

from pytorchstudy.insect_snd_sep.prepare_data import (
    prepare_librimix,
)


app = typer.Typer()


@app.command(help="prepare sound separation dataset")
def prepare(
    datapath: pathlib.Path = typer.Option(
        # default=...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="input dir of dataset",
    ),
    savepath: pathlib.Path = typer.Option(
        # default=...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="output dir of dataset",
    ),
    n_spks: int = 2,
    fs: int = 8000,
    version: Annotated[str, typer.Option()] = "wav8k/min/",
    set_types: Annotated[Tuple[str, str, str], typer.Option()] = (
        "train-360",
        "dev",
        "test",
    ),
):
    print(f"datapath: {datapath}")
    print(f"saveapth: {savepath}")
    print(f"n_spks: {n_spks}")
    print(f"fs: {fs}")
    print(f"version: {version}")
    print(f"set_types: {set_types}")

    prepare_librimix(
        datapath=str(datapath),
        savepath=str(savepath),
        n_spks=n_spks,
        skip_prep=False,
        librimix_addnoise=False,
        fs=fs,
        version=version,
        set_types=set_types,
    )


@app.command(help="sepformer inference")
def inference(
    test_sample: pathlib.Path = typer.Argument(
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="test sample file",
    ),
    out_dir: pathlib.Path = typer.Argument(
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="output folder for estimated sources",
    ),
    model_dir: Optional[pathlib.Path] = typer.Option(
        exists=True,
        file_okay=False,
        dir_okay=True,
        default=None,
        help="model folder",
    ),
    model_name: str = typer.Option(),
    out_sr: int = 44100,
    n_src: int = 2,
    cuda: bool = False,
):
    # https://huggingface.co/speechbrain/sepformer-libri3mix
    from speechbrain.pretrained import SepformerSeparation as separator
    import soundfile as sf

    if model_dir:
        model_t = str(model_dir)
    else:
        model_t = model_name

    if cuda:
        run_opts = {"device": "cuda"}
    else:
        run_opts = {}

    model = separator.from_hparams(source=model_t, run_opts=run_opts)
    est_sources = model.separate_file(path=str(test_sample))

    print(est_sources.shape)

    sample_stem = test_sample.stem

    print("\nseparated source 1")
    est_src_1 = est_sources[:, :, 0].detach().cpu().squeeze()
    sf.write(out_dir / f"{sample_stem}_est_s1.wav", est_src_1, out_sr)

    print("\nseparated source 2")
    est_src_2 = est_sources[:, :, 1].detach().cpu().squeeze()
    sf.write(out_dir / f"{sample_stem}_est_s2.wav", est_src_2, out_sr)

    if n_src == 3:
        print("\nseparated source 3")
        est_src_3 = est_sources[:, :, 2].detach().cpu().squeeze()
        sf.write(out_dir / f"{sample_stem}_est_s3.wav", est_src_3, out_sr)


@app.command(help="enhance speech")
def enhance(
    test_sample: pathlib.Path = typer.Argument(
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="test sample file",
    ),
    out_dir: pathlib.Path = typer.Argument(
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="output folder for estimated sources",
    ),
    model_dir: Optional[pathlib.Path] = typer.Option(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        # default=None,
        help="model folder",
    ),
    model_name: str = typer.Option(),
    out_sr: int = 44100,
    cuda: bool = False,
):
    # https://huggingface.co/speechbrain/sepformer-dns4-16k-enhancement
    from speechbrain.pretrained import SepformerSeparation as separator
    import torchaudio

    if model_dir:
        model_t = str(model_dir)
    else:
        model_t = model_name

    if cuda:
        run_opts = {"device": "cuda"}
    else:
        run_opts = {}

    model = separator.from_hparams(source=model_t, run_opts=run_opts)
    est_sources = model.separate_file(path=str(test_sample))

    print(est_sources.shape)

    sample_stem = test_sample.stem

    print("\nseparated source 1")
    est_src_1 = est_sources[:, :, 0].detach().cpu()
    torchaudio.save(out_dir / f"{sample_stem}_est_s1.wav", est_src_1, out_sr)


@app.command(help="compute audio embedding")
def audio_embedding(
    test_sample: pathlib.Path = typer.Option(
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="test sample file",
    ),
    model: str = "speechbrain/spkrec-ecapa-voxceleb",
    hparams_fn: str = "hyperparams.yaml",
):
    from pytorchstudy.insect_snd_id.verify_speaker import compute_embedding

    print(f"model: {model}")
    em = compute_embedding(model, hparams_fn, test_sample)
    print(f"embedding:\n{em}")


@app.command(help="classify audio")
def audio_classify(
    test_sample: Annotated[
        pathlib.Path,
        typer.Option(
            exists=True,
            file_okay=True,
            dir_okay=False,
            help="test sample file",
        ),
    ],
    model: str = "speechbrain/spkrec-ecapa-voxceleb",
    hparams_fn: str = "hyperparams.yaml",
):
    from pytorchstudy.insect_snd_id.verify_speaker import audio_classify as ac

    print(f"model: {model}")
    out_prob, score, index, text_lab = ac(model, hparams_fn, test_sample)
    print(
        f"out prob:\n{out_prob}, score: {score}, index: {index}, text label:{text_lab}"
    )


@app.command(help="verify speaker")
def verify(
    audio_1: pathlib.Path = typer.Option(
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="test sample file",
    ),
    audio_2: pathlib.Path = typer.Option(
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="test sample file",
    ),
    # savedir: Annotated[
    #     pathlib.Path,
    #     typer.Option(
    #         exists=True,
    #         file_okay=False,
    #         dir_okay=True,
    #         help="dir to save model",
    #     ),
    # ],
    model: str = "speechbrain/spkrec-ecapa-voxceleb",
    hparams_fn: str = "hyperparams.yaml",
):
    from pytorchstudy.insect_snd_id.verify_speaker import verify_2 as sv

    score, pred = sv(model, hparams_fn, audio_1, audio_2)
    print(f"score: {score}, predication: {pred}")


@app.command(help="classify insects")
def cls_insects(
    test_csv: pathlib.Path = typer.Option(
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="test csv file",
    ),
    data_folder: pathlib.Path = typer.Option(
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="data folder of wav files",
    ),
    model: str = "speechbrain/spkrec-ecapa-voxceleb",
    hparams_fn: str = "hyperparams.yaml",
):
    import csv
    import functional as pyfun
    from pytorchstudy.insect_snd_id.verify_speaker import insect_cls

    data = []
    with open(test_csv, "r") as csvfile:
        csvreader = csv.DictReader(
            csvfile
        )  # Use DictReader to handle header automatically

        for row in csvreader:
            data.append((row["spk_id"], row["wav"]))
    # print(data)

    data = (
        pyfun.seq(data)
        .map(lambda t: (t[0], t[1].replace("$data_root", str(data_folder))))
        .group_by_key()
        .to_dict()
    )
    logger.debug(data)

    # ret = Result.do(
    #     same_set + diff_set
    #     for test_data in test_set(data)
    #     for diff_set in diff_gh_test_set(this_gh, data)
    # )
    ret = insect_cls(model, hparams_fn, data)
    match ret:
        case Success(tl):
            print(tl)
        case Failure(ex):
            print(str(ex))


if __name__ == "__main__":
    print(f"python version is {sys.version_info}")
    if not (sys.version_info.major == 3 and sys.version_info.minor >= 9):
        sys.exit("this program needs python 3.9 and above to run")

    # https://towardsdatascience.com/a-simple-guide-to-command-line-arguments-with-argparse-6824c30ab1c3
    print(f"sys.path:\n{sys.path}")

    # l_fmt = '[%(levelname)s] %(asctime)s - %(message)s'
    # logging.basicConfig(level=logging.ERROR, format=l_fmt)

    # https://medium.com/pythoneers/master-logging-in-python-73cd2ff4a7cb
    # https://medium.com/@techfossguru/ultimate-guide-to-learn-python-logging-module-in-depth-tutorial-6f7e642e73e1
    # https://medium.com/@cyberdud3/a-step-by-step-guide-to-configuring-python-logging-with-yaml-files-914baea5a0e5
    with open("logging_conf.yaml", "rt") as f:
        config = yaml.safe_load(f.read())
    logging.config.dictConfig(config)
    logger = logging.getLogger("development")
    # logger.setLevel(logging.ERROR)

    app()
