import sys
import pathlib
import typer
import logging


app = typer.Typer()


@app.command(help="confusion matrix")
def conf_matrix(
    type: str = "binaray",
):
    from torch import tensor
    from torchmetrics import ConfusionMatrix

    target = tensor([1, 1, 0, 0])
    preds = tensor([0, 1, 0, 0])
    confmat = ConfusionMatrix(task="binary", num_classes=2)
    confmat(preds, target)
    # fig, ax = confmat.plot()

    from torchmetrics.classification import BinaryConfusionMatrix

    target = tensor([1, 1, 0, 0])
    preds = tensor([0.35, 0.85, 0.48, 0.01])
    bcm = BinaryConfusionMatrix()
    bcm(preds, target)
    # fig, ax = bcm.plot()

    from torch import randint
    from torchmetrics.classification import MulticlassConfusionMatrix

    metric = MulticlassConfusionMatrix(num_classes=5)
    metric.update(randint(5, (20,)), randint(5, (20,)))
    fig_, ax_ = metric.plot()


@app.command(help="nvidia gpu info")
def nv_gpu():
    import torch

    device_count = torch.cuda.device_count()
    if device_count > 0:
        print(f"Number of CUDA devices available: {device_count}")
        for device_idx in range(device_count):
            device_name = torch.cuda.get_device_name(device_idx)
            print(f"Device {device_idx}: {device_name}")
    else:
        print("No CUDA devices found.")

    # Specify the device type
    torch.cuda.is_available()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)


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
