# Explain prepare_data.py

## How to run

```sh
PYTHONPATH=. python3 app/snd_sep.py prepare --help

PYTHONPATH=. python3 app/snd_sep.py prepare \
--datapath /media/ml-data/projects/ml/datasets/sound/insect/ \
--savepath ~/tmp/ml --n_spks 3 --fs 16000 \
--set_types train-100 dev test
```

The script will generate 3 CSV files, for instance train_mix_3.csv, val_mix_3.csv,
test_mix_3.csv to '--savepath' and copy the sound tracks over in train/val/test folders:

```text
├── test
├── test_mix_3.csv
├── train
├── train_mix_3.csv
├── val
└── val_mix_3.csv
```

```text
ID,mix_wav,s1_wav,s2_wav,s3_wav,noise_wav
0,$data_root/train/mix/mix_0.wav,$data_root/train/s1/XC751400 - Lesser Marsh Grasshopper - Chorthippus albomarginatus_mono_4.wav,$data_root/train/s2/XC501261_mono_44100_denoised_14.wav,$data_root/train/s3/cricket1_71.wav,$data_root/train/noise/drone-only-mono-44100_12.wav
1,$data_root/train/mix/mix_1.wav,$data_root/train/s1/XC751377 - Lesser Marsh Grasshopper - Chorthippus albomarginatus_mono_4.wav,$data_root/train/s2/XC134502_mono_44100_denoised_2.wav,$data_root/train/s3/cricket1_96.wav,$data_root/train/noise/drone-with-dir-mic_mono_44100_13.wav
2,$data_root/train/mix/mix_2.wav,$data_root/train/s1/XC751376 - Lesser Marsh Grasshopper - Chorthippus albomarginatus_mono_7.wav,$data_root/train/s2/XC54018_mono_44100_denoised_24.wav,$data_root/train/s3/cricket1_67.wav,$data_root/train/noise/drone-only-mono-44100_1.wav
```

train_mix
Among the columns, what the train.py really cares are:
ID, mix_wav, s1_wav, s2_wav, s3_wav if mix3, noise_wav if use_wham_noise.

columns duration, mix_wav_format, mix_wav_opts, s1_wav_format, s1_mix_wav_opts, s2_wav_format, s2_wav_opts, s3_wav_format, mix_wav_opts, noise_wav_format, noise_wav_opts are optional.

libri3mix_dev.csv

```csv
ID,duration,mix_wav,mix_wav_format,mix_wav_opts,s1_wav,s1_wav_format,s1_wav_opts,s2_wav,s2_wav_format,s2_wav_opts,s3_wav,s3_wav_format,s3_wav_opts,noise_wav,noise_wav_format,noise_wav_opts
0,1.0,/media/ml-data/projects/ml/datasets/sound/LibriMix/dataset/Libri3Mix/wav16k/min/dev/mix_clean/6313-66125-0003_1673-143396-0014_6345-93302-0027.wav,wav,,/media/ml-data/projects/ml/datasets/sound/LibriMix/dataset/Libri3Mix/wav16k/min/dev/s1/6313-66125-0003_1673-143396-0014_6345-93302-0027.wav,wav,,/media/ml-data/projects/ml/datasets/sound/LibriMix/dataset/Libri3Mix/wav16k/min/dev/s2/6313-66125-0003_1673-143396-0014_6345-93302-0027.wav,wav,,/media/ml-data/projects/ml/datasets/sound/LibriMix/dataset/Libri3Mix/wav16k/min/dev/s3/6313-66125-0003_1673-143396-0014_6345-93302-0027.wav,wav,,/media/ml-data/projects/ml/datasets/sound/LibriMix/dataset/Libri3Mix/wav16k/min/dev/noise/6313-66125-0003_1673-143396-0014_6345-93302-0027.wav,wav,
1,1.0,/media/ml-data/projects/ml/datasets/sound/LibriMix/dataset/Libri3Mix/wav16k/min/dev/mix_clean/1919-142785-0054_777-126732-0057_3752-4944-0042.wav,wav,,/media/ml-data/projects/ml/datasets/sound/LibriMix/dataset/Libri3Mix/wav16k/min/dev/s1/1919-142785-0054_777-126732-0057_3752-4944-0042.wav,wav,,/media/ml-data/projects/ml/datasets/sound/LibriMix/dataset/Libri3Mix/wav16k/min/dev/s2/1919-142785-0054_777-126732-0057_3752-4944-0042.wav,wav,,/media/ml-data/projects/ml/datasets/sound/LibriMix/dataset/Libri3Mix/wav16k/min/dev/s3/1919-142785-0054_777-126732-0057_3752-4944-0042.wav,wav,,/media/ml-data/projects/ml/datasets/sound/LibriMix/dataset/Libri3Mix/wav16k/min/dev/noise/1919-142785-0054_777-126732-0057_3752-4944-0042.wav,wav,
```

libri3mix_test.csv

```csv
ID,duration,mix_wav,mix_wav_format,mix_wav_opts,s1_wav,s1_wav_format,s1_wav_opts,s2_wav,s2_wav_format,s2_wav_opts,s3_wav,s3_wav_format,s3_wav_opts,noise_wav,noise_wav_format,noise_wav_opts
0,1.0,/media/ml-data/projects/ml/datasets/sound/LibriMix/dataset/Libri3Mix/wav16k/min/dev/mix_clean/6313-66125-0003_1673-143396-0014_6345-93302-0027.wav,wav,,/media/ml-data/projects/ml/datasets/sound/LibriMix/dataset/Libri3Mix/wav16k/min/dev/s1/6313-66125-0003_1673-143396-0014_6345-93302-0027.wav,wav,,/media/ml-data/projects/ml/datasets/sound/LibriMix/dataset/Libri3Mix/wav16k/min/dev/s2/6313-66125-0003_1673-143396-0014_6345-93302-0027.wav,wav,,/media/ml-data/projects/ml/datasets/sound/LibriMix/dataset/Libri3Mix/wav16k/min/dev/s3/6313-66125-0003_1673-143396-0014_6345-93302-0027.wav,wav,,/media/ml-data/projects/ml/datasets/sound/LibriMix/dataset/Libri3Mix/wav16k/min/dev/noise/6313-66125-0003_1673-143396-0014_6345-93302-0027.wav,wav,
1,1.0,/media/ml-data/projects/ml/datasets/sound/LibriMix/dataset/Libri3Mix/wav16k/min/dev/mix_clean/1919-142785-0054_777-126732-0057_3752-4944-0042.wav,wav,,/media/ml-data/projects/ml/datasets/sound/LibriMix/dataset/Libri3Mix/wav16k/min/dev/s1/1919-142785-0054_777-126732-0057_3752-4944-0042.wav,wav,,/media/ml-data/projects/ml/datasets/sound/LibriMix/dataset/Libri3Mix/wav16k/min/dev/s2/1919-142785-0054_777-126732-0057_3752-4944-0042.wav,wav,,/media/ml-data/projects/ml/datasets/sound/LibriMix/dataset/Libri3Mix/wav16k/min/dev/s3/1919-142785-0054_777-126732-0057_3752-4944-0042.wav,wav,,/media/ml-data/projects/ml/datasets/sound/LibriMix/dataset/Libri3Mix/wav16k/min/dev/noise/1919-142785-0054_777-126732-0057_3752-4944-0042.wav,wav,
```

libri3mix_train-100.csv

```csv
ID,duration,mix_wav,mix_wav_format,mix_wav_opts,s1_wav,s1_wav_format,s1_wav_opts,s2_wav,s2_wav_format,s2_wav_opts,s3_wav,s3_wav_format,s3_wav_opts,noise_wav,noise_wav_format,noise_wav_opts
0,1.0,/media/ml-data/projects/ml/datasets/sound/LibriMix/dataset/Libri3Mix/wav16k/min/train-100/mix_clean/3436-172171-0055_730-360-0044_2989-138028-0071.wav,wav,,/media/ml-data/projects/ml/datasets/sound/LibriMix/dataset/Libri3Mix/wav16k/min/train-100/s1/3436-172171-0055_730-360-0044_2989-138028-0071.wav,wav,,/media/ml-data/projects/ml/datasets/sound/LibriMix/dataset/Libri3Mix/wav16k/min/train-100/s2/3436-172171-0055_730-360-0044_2989-138028-0071.wav,wav,,/media/ml-data/projects/ml/datasets/sound/LibriMix/dataset/Libri3Mix/wav16k/min/train-100/s3/3436-172171-0055_730-360-0044_2989-138028-0071.wav,wav,,/media/ml-data/projects/ml/datasets/sound/LibriMix/dataset/Libri3Mix/wav16k/min/train-100/noise/3436-172171-0055_730-360-0044_2989-138028-0071.wav,wav,
1,1.0,/media/ml-data/projects/ml/datasets/sound/LibriMix/dataset/Libri3Mix/wav16k/min/train-100/mix_clean/6019-3185-0037_2289-152257-0014_2989-138028-0037.wav,wav,,/media/ml-data/projects/ml/datasets/sound/LibriMix/dataset/Libri3Mix/wav16k/min/train-100/s1/6019-3185-0037_2289-152257-0014_2989-138028-0037.wav,wav,,/media/ml-data/projects/ml/datasets/sound/LibriMix/dataset/Libri3Mix/wav16k/min/train-100/s2/6019-3185-0037_2289-152257-0014_2989-138028-0037.wav,wav,,/media/ml-data/projects/ml/datasets/sound/LibriMix/dataset/Libri3Mix/wav16k/min/train-100/s3/6019-3185-0037_2289-152257-0014_2989-138028-0037.wav,wav,,/media/ml-data/projects/ml/datasets/sound/LibriMix/dataset/Libri3Mix/wav16k/min/train-100/noise/6019-3185-0037_2289-152257-0014_2989-138028-0037.wav,wav,
```

