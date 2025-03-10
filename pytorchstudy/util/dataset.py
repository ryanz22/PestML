import csv
from pathlib import Path
import shutil
from typing import Any
import random
import functional as pyfn
from returns.result import Result, safe, Failure, Success


def pick_from_ds(
    in_dir: Path, out_dir: Path, csv_file: str, sample_id: int
) -> Result[None, Exception]:
    ds_type_list = ["train", "val", "test"]
    ret = pyfn.seq(ds_type_list).filter(lambda s: s in Path(csv_file).stem)
    if ret.empty():
        return Failure(
            Exception(f"csv file name {csv_file} doesn't contain any of {ds_type_list}")
        )

    ds_type = ret.first(no_wrap=True)
    print(f"This is {ds_type} meta csv file")

    with open(in_dir / csv_file, "r") as csv_f:
        reader = csv.DictReader(csv_f)
        print(f"type of csv dictReader: {type(reader)}")
        header = reader.fieldnames
        print(f"header: {header}")

        rows = list(reader)
        total = len(rows)
        print(f"{csv_file} has total {total} rows")

        if sample_id >= total:
            return Failure(Exception(f"sample-id {sample_id} is out of scope"))

        # s2 = rows[0]["s2_wav"]
        # print(f"source 2 wav: {s2}")
        row = rows[sample_id]
        # print(f"row {sample_id}:\n{row}")
        return process_row(row, in_dir, out_dir, sample_id)

    return Success(None)


def process_row(
    row: dict[str, str], in_dir: Path, out_dir: Path, sample_id: int
) -> Result[None, Exception]:
    def handle(
        key: str, val: str, in_dir: Path, out_dir: Path, sample_id: int
    ) -> None | Exception:
        if len(val.strip()) == 0:  # ignore empty value
            return None

        match key:
            case "mix_wav" | "s1_wav" | "s2_wav" | "s3_wav" | "noise_wav":
                print(f"{key}: {val}\n")
                return process_item(key, val, in_dir, out_dir, sample_id)
            case "ID":
                print(f"ID: {val}\n")
                return None
            case _:
                return Exception(f"Unexpected column: {key}")

    ret = (
        pyfn.seq(list(row.items()))
        .map(lambda t: handle(t[0], t[1], in_dir, out_dir, sample_id))
        .filter(lambda r: isinstance(r, Exception))
        .list()
    )
    print(ret)

    if pyfn.seq(ret).empty():
        return Success(None)
    else:
        return Failure(Exception(pyfn.seq(ret).make_string("\n")))


def process_item(
    key: str, val: str, in_dir: Path, out_dir: Path, sample_id: int
) -> None | Exception:
    sound_type = key.replace("_wav", "")
    fn = val.replace("$data_root/", "")
    full_path = in_dir / fn
    print(full_path)

    if full_path.is_symlink():
        print(f"{full_path} is a symbolic link and need to do resolving")
        full_path = full_path.resolve()

    if not (full_path.exists() and full_path.is_file()):
        msg = f"path doesn't exist or not a file: {full_path}"
        # print(msg)
        return Exception(msg)

    match sound_type:
        case "mix":
            out_path = out_dir / full_path.name
        case _:
            out_path = out_dir / f"mix_{sample_id}_{sound_type}.wav"

    print(f"out path: {out_path}")

    shutil.copy(full_path, out_path)

    return None


def random_pick_except_me(fl: list, except_idx: int) -> None | tuple[Any, int]:
    cnt = len(fl)
    if cnt == 0:
        return None

    rand = random.randint(0, cnt - 1)
    while rand == except_idx:
        rand = random.randint(0, cnt - 1)

    return fl[rand], rand
