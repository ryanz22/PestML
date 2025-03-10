import logging
import random
from pathlib import Path
import torchaudio
import torch
from speechbrain.pretrained import EncoderClassifier
from speechbrain.pretrained import SpeakerRecognition
import functional as pyfn
from returns.result import Result, safe, Success, Failure

from pytorchstudy.util.dataset import random_pick_except_me


prod_logger = logging.getLogger("production")
dev_logger = logging.getLogger("development")


def compute_embedding(model: str, hp_fn: str, fp: Path):
    classifier = EncoderClassifier.from_hparams(
        source=model,
        hparams_file=hp_fn,
        # savedir="pretrained_models/spkrec-ecapa-voxceleb",
        run_opts={"device": "cuda"},
    )
    signal, fs = torchaudio.load(fp)
    embeddings = classifier.encode_batch(signal)

    return embeddings


def audio_classify(model: str, hp_fn: str, fp: Path):
    classifier = EncoderClassifier.from_hparams(
        source=model,
        hparams_file=hp_fn,
        # savedir="pretrained_models/spkrec-ecapa-voxceleb",
        run_opts={"device": "cuda"},
    )
    out_prob, score, index, text_lab = classifier.classify_file(str(fp))

    return out_prob, score, index, text_lab


def verify(model: str, hp_fn: str, fp1: Path, fp2: Path):
    # source : str or FetchSource
    #     Where to look for the file. This is interpreted in special ways:
    #     First, if the source begins with "http://" or "https://", it is
    #     interpreted as a web address and the file is downloaded.
    #     Second, if the source is a valid directory path, a symlink is
    #     created to the file.
    #     Otherwise, the source is interpreted as a Huggingface model hub ID, and
    #     the file is downloaded from there.
    verification = SpeakerRecognition.from_hparams(
        source=model,
        hparams_file=hp_fn,
        # savedir=savedir,
        run_opts={"device": "cuda"},
    )
    score, prediction = verification.verify_files(str(fp1), str(fp2))

    return score, prediction


def verify_2(model: str, hp_fn: str, fp1: Path, fp2: Path):
    emb1 = compute_embedding(model, hp_fn, fp1)
    emb2 = compute_embedding(model, hp_fn, fp2)
    similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
    score = similarity(emb1, emb2)

    threshold = 0.25
    return score, score > threshold


@safe
# def insect_verify(
#     model: str, hp_fn: str, dataset: list[tuple[bool, tuple[str, str], tuple[str, str]]]
# ) -> tuple[list, list]:
def insect_cls(model: str, hp_fn: str, data: dict[str, list[str]]) -> tuple[list, list]:
    dev_logger.info("Insect classify starts ...")

    classifier = EncoderClassifier.from_hparams(
        source=model,
        hparams_file=hp_fn,
        # savedir="pretrained_models/spkrec-ecapa-voxceleb",
        run_opts={"device": "cuda"},
    )

    # waveform = self.load_audio(path)
    # waveform, _ = torchaudio.load(str(path), channels_first=False)
    # emb = self.encode_batch(wavs, wav_lens)
    # out_prob = self.mods.classifier(emb).squeeze(1)
    # score, index = torch.max(out_prob, dim=-1)
    # text_lab = self.hparams.label_encoder.decode_torch(index)

    for gh, wav_list in data.items():
        dev_logger.info(f"process {gh} which has {len(wav_list)} sound tracks")
        # same gh
        assert len(wav_list) > 1, f"grasshopper [{gh}] has less than two sound tracks"

        wavs = (
            pyfn.seq(wav_list)
            .map(lambda fn: torchaudio.load(fn))
            .map(lambda t: t[0])
            .to_list()
        )
        dev_logger.debug(f"wavs len {len(wavs)}")

        # def classify_batch(self, wavs, wav_lens=None):
        out_prob, score, index, text_lab = classifier.classify_batch(torch.stack(wavs))
        # dev_logger.debug(f"out_prob: {out_prob}")
        # dev_logger.debug(f"score: {score}")
        # dev_logger.debug(f"index: {index}")
        # dev_logger.debug(f"text_lab: {text_lab}")

        print(f"out_prob: {out_prob}")
        print(f"score: {score}")
        print(f"index: {index}")
        print(f"text_lab: {text_lab}")

    dev_logger.info("Insect classify ended ...")

    return [], []


def test_set(
    data: dict[str, list[str]]
) -> Result[list[tuple[bool, tuple[str, str], tuple[str, str]]], Exception]:
    if not data:
        return Failure(Exception("empty data"))

    ret = []
    for gh, wav_list in data.items():
        # same gh
        if len(wav_list) < 2:
            Failure(Exception(f"grasshopper [{gh}] has less than two sound tracks"))

        same_set = same_gh_test_set(gh, wav_list)
        diff_set = diff_gh_test_set(gh, data)

        ret.append(same_set + diff_set)

    return Success(ret)


def same_gh_test_set(
    gh: str, wav_fn_list: list[str]
) -> list[tuple[bool, tuple[str, str], tuple[str, str]]]:
    assert len(wav_fn_list) > 1, "list must have two or more items"

    ret = []
    next_idx = 0
    l_len = len(wav_fn_list)
    for idx in range(0, l_len - 1):
        el = wav_fn_list[idx]
        next_idx = idx + 1

        for tmp in range(next_idx, l_len):
            el2 = wav_fn_list[tmp]
            ret.append((True, (gh, el), (gh, el2)))

    return ret


def diff_gh_test_set(
    gh: str, data: dict[str, list[str]]
) -> list[tuple[bool, tuple[str, str], tuple[str, str]]]:
    ret = []

    l = data[gh]
    assert len(l) > 1, f"{gh} list must have two or more items"
    keys = set(data.keys())
    keys.remove(gh)

    for el in l:
        r_key = random.choice(list(keys))
        l2 = data[r_key]
        assert len(l2) > 1, f"{r_key} list must have two or more items"
        el2 = random.choice(l2)
        ret.append((False, (gh, el), (r_key, el2)))

    return ret
