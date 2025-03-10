# Speech separation with LibriMix

This folder contains some popular recipes for the [LibriMix Dataset](https://arxiv.org/pdf/2005.11262.pdf) (2/3 sources).

* This recipe supports train with several source separation models on LibriMix, including [Sepformer](https://arxiv.org/abs/2010.13154), [DPRNN](https://arxiv.org/abs/1910.06379), [ConvTasnet](https://arxiv.org/abs/1809.07454), [DPTNet](https://arxiv.org/abs/2007.13975).

## How to setup

Additional dependencies:
```
pip install mir_eval
pip install pyloudnorm
```

## How to run
To run it:

```shell
python train.py hparams/sepformer-libri2mix.yaml --data_folder yourpath/Libri2Mix
python train.py hparams/sepformer-libri3mix.yaml --data_folder yourpath/Libri3Mix
```

Run on 3090/4090

```shell
PYTHONPATH=. nohup python3 \
pytorchstudy/insect_snd_sep/train_insect.py \
config/insect_snd_sep/hparams/insect-mix2-gh-gh-clean.yaml \
--data_folder datasets/pestdataprocess/sound/insect_sep_ds/train-ds/mix2-gh-gh-clean/ &> mix2-gh-gh-clean_nohup.txt &
```

Run on A100-80GB

```shell
PYTHONPATH=. CUDA_VISIBLE_DEVICES=3 nohup python3 \
pytorchstudy/insect_snd_sep/train_insect.py \
config/insect_snd_sep/hparams/insect-mix2-gh-gh-clean.yaml \
--data_folder /media/zhangjw/ml-data/projects/ml/datasets/pestdataprocess/sound/insect_sep_ds/train-ds/mix2-gh-gh-clean/ &> mix2-gh-gh-clean_nohup.txt &
```

How to run multiple GPUs:

  "To use data_parallel backend, start your script with:\n\t"
                "python experiment.py hyperparams.yaml "
                "--data_parallel_backend=True"
                "To use DDP backend, start your script with:\n\t"
                "python -m torch.distributed.lunch [args]\n"
                "experiment.py hyperparams.yaml --distributed_launch=True "
                "--distributed_backend=nccl"


Note that during training we print the negative SI-SNR (as we treat this value as the loss).

### Inference

```shell
PYTHONPATH=. python app/snd_sep.py inference --test-sample data/test/mix3-gh-cricket-bird-clean/mix_0.wav --model-dir models/mix3-gh-cricket-bird-clean-k8s4/ --out-dir data/test/mix3-gh-cricket-bird-clean/ --n-src 3

PYTHONPATH=. python app/snd_sep.py inference --test-sample ext-data/test/mix2-gh-cricket-aug-clean/mix_4.wav --model-dir ext-models/convtasnet/mix2-gh-cricket-aug-clean-k16s8/ --out-dir ext-data/test/mix2-gh-cricket-aug-clean/cnvtasnet/k16s8/ --n-src 2
```

### Results

Here are the SI - SNRi results (in dB) on the test set of LibriMix dataset with SepFormer:

| | SepFormer. Libri2Mix |
| --- | --- |
|SpeedAugment | 20.1|
|DynamicMixing | 20.4|


| | SepFormer. Libri3Mix |
| --- | --- |
|SpeedAugment | 18.4|
|DynamicMixing | 19.0|


### Example calls for running the training scripts

* Libri2Mix with dynamic mixing `python train.py hparams/sepformer-libri2mix.yaml --data_folder yourpath/Libri2Mix/ --base_folder_dm yourpath/LibriSpeech_processed --dynamic_mixing True`

* Libri3Mix with dynamic mixing `python train.py hparams/sepformer-libri3mix.yaml --data_folder yourpath/Libri3Mix/ --base_folder_dm yourpath/LibriSpeech_processed --dynamic_mixing True`

* Libri2Mix with dynamic mixing with WHAM! noise in the mixtures `python train.py hparams/sepformer-libri2mix.yaml --data_folder yourpath/Libri2Mix/ --base_folder_dm yourpath/LibriSpeech_processed --dynamic_mixing True --use_wham_noise True`

* Libri3Mix with dynamic mixing with WHAM! noise in the mixtures `python train.py hparams/sepformer-libri3mix.yaml --data_folder yourpath/Libri3Mix/ --base_folder_dm yourpath/LibriSpeech_processed --dynamic_mixing True --use_wham_noise True`


The output folder with the trained model and the logs can be found [here](https://drive.google.com/drive/folders/1DN49LtAs6cq1X0jZ8tRMlh2Pj6AecClz?usp=sharing) for 3-speaker mixtures and [here](https://drive.google.com/drive/folders/1NPTXw4i9Vmahhr5BSQQa-ZTTm45FwYJA?usp=sharing) for 2-speakers ones.

### Multi-GPU training

You can run the following command to train the model using Distributed Data Parallel (DDP) with 2 GPUs:

**NOTE** nohup has problem when working with distributed training. The training process will exit
when terminal logout.

[torchrun](https://pytorch.org/docs/stable/elastic/run.html)

**batch size should be same as single GPU**

```shell
PYTHONPATH=. CUDA_VISIBLE_DEVICES=2,3 nohup torchrun --standalone --nnodes=1 \
--nproc_per_node=2 pytorchstudy/insect_snd_sep/train_insect.py \
config/insect_snd_sep/hparams/insect-mix2-gh-gh-aug-clean-k8s4.yaml \
--data_folder data/sound/insect_sep_ds/train-ds/mix2-gh-gh-aug-clean/ \
--distributed_backend='nccl' &> mix2-gh-gh-clean-k8s4.out &
```

You can add the other runtime options as appropriate. For more complete information on multi-GPU usage, take a look at this [tutorial](https://colab.research.google.com/drive/13pBUacPiotw1IvyffvGZ-HrtBr9T6l15?usp=sharing).


https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html

https://pytorch.org/tutorials/intermediate/model_parallel_tutorial.html

PyTorch Distributed: Experiences on Accelerating
Data Parallel Training

http://www.vldb.org/pvldb/vol13/p3005-li.pdf

https://pytorch.org/tutorials/beginner/dist_overview.html

**explain more about following:**

The problem lies in the command nohup, the distributed training process operated by nohup will receive the above SIGHUP signal when closing the terminal, even if we specify the command nohup. Switch to tmux will resolve this issue.
ChatGPT
In the context provided, the issue revolves around running a distributed training process using the nohup command. nohup is a Unix command that stands for "no hang-up," and it is used to run a command in the background, allowing it to continue running even if the terminal session that started it is closed or disconnected.

However, there is a problem with using nohup in this scenario. When the terminal session is closed, it sends a signal called SIGHUP (hang-up signal) to all the processes associated with that terminal, including the distributed training process running with nohup. Receiving this signal typically terminates the process, even if nohup was used to detach it from the terminal.

To overcome this issue, the suggestion is to use a tool called tmux. tmux is a terminal multiplexer that allows you to create multiple terminal sessions within a single terminal window or SSH session. It provides a way to keep processes running even if the terminal is closed or disconnected.

Here's how tmux can help resolve the problem with the distributed training process:

Start a new tmux session in the terminal.
Run the command for distributed training within the tmux session.
Detach the tmux session: You can do this by pressing Ctrl + B followed by d. This action will leave the tmux session running in the background.
Close or disconnect from the terminal: Now, even if you close the terminal or disconnect from the SSH session, the tmux session and the distributed training process within it will continue running.
If you need to reattach to the tmux session later to check on the training process or interact with it, you can do so by opening a new terminal or SSH session and running:

tmux attach-session

This will reattach you to the existing tmux session with the running distributed training process.

Using tmux ensures that the distributed training process remains persistent even when the controlling terminal is closed or disconnected, making it a suitable solution for long-running tasks like distributed training that require a stable environment.

## Datasets

### Libri2/3 Mix
* The Dataset can be created using the scripts at `https://github.com/JorisCos/LibriMix`.


### Dynamic Mixing

* This recipe supports dynamic mixing where the training data is dynamically created in order to obtain new utterance combinations during training.


## **About SpeechBrain**
- Website: https://speechbrain.github.io/
- Code: https://github.com/speechbrain/speechbrain/
- HuggingFace: https://huggingface.co/speechbrain/


## **Citing SpeechBrain**
Please, cite SpeechBrain if you use it for your research or business.

```bibtex
@misc{speechbrain,
  title={{SpeechBrain}: A General-Purpose Speech Toolkit},
  author={Mirco Ravanelli and Titouan Parcollet and Peter Plantinga and Aku Rouhe and Samuele Cornell and Loren Lugosch and Cem Subakan and Nauman Dawalatabad and Abdelwahab Heba and Jianyuan Zhong and Ju-Chieh Chou and Sung-Lin Yeh and Szu-Wei Fu and Chien-Feng Liao and Elena Rastorgueva and François Grondin and William Aris and Hwidong Na and Yan Gao and Renato De Mori and Yoshua Bengio},
  year={2021},
  eprint={2106.04624},
  archivePrefix={arXiv},
  primaryClass={eess.AS},
  note={arXiv:2106.04624}
}
```

