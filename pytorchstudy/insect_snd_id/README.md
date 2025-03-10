# Insect audio identification

## How to compute embedding

```sh
PYTHONPATH=. python app/snd_sep.py audio-embedding \
--test-sample ext-gh-cls/grasshopper-aug-ds/gh-4/gh-4_mono_44k_denoised_1_9.wav \
--model ext-models/audio-id/ecapa/ --hparams-fn hyperparams_train.yaml
```

## How to check similarity

```sh
PYTHONPATH=. python app/snd_sep.py verify \
--audio-1 ext-gh-cls/grasshopper-aug-ds/gh-4/gh-4_mono_44k_denoised_1_9.wav \
--audio-2 ext-gh-cls/grasshopper-aug-ds/gh-4/gh-4_mono_44k_denoised_1_9.wav \
--model ext-models/audio-id/ecapa/ --hparams-fn hyperparams_train.yaml
```

## How to classify audio

```sh

```

## How to train

Single GPU and foreground

```sh
PYTHONPATH=. CUDA_VISIBLE_DEVICES=3 python3 \
pytorchstudy/insect_snd_id/train_speaker_embeddings.py \
config/insect_id/hparams/train_ecapa_tdnn-nmels160.yaml \
--data_folder data/sound/insect_id/grasshopper-aug-ds
```

Single GPU and background

```sh
PYTHONPATH=. CUDA_VISIBLE_DEVICES=3 nohup python3 \
pytorchstudy/insect_snd_id/train_speaker_embeddings.py \
config/insect_id/hparams/train_ecapa_tdnn-nmels160.yaml \
--data_folder data/sound/insect_id/grasshopper-aug-ds \
&> ecapa-tdnn-gh-aug-ds-nmels160.log &
```

Multiple GPU and background

start tmux
create a new session
run train script
detach by "ctrl-b d"
list tmux session by tmux ls
attach session by tmux a -s name

```sh
PYTHONPATH=. CUDA_VISIBLE_DEVICES=2,3 python3 -m torch.distributed.launch \ --nproc_per_node=2 pytorchstudy/insect_snd_id/train_speaker_embeddings.py \
config/insect_id/hparams/train_ecapa_tdnn-nmels160.yaml \
--data_folder data/sound/insect_id/grasshopper-aug-ds \ --distributed_launch --distributed_backend='nccl' \
&> ecapa-tdnn-gh-aug-ds-nmels160.log &
```
