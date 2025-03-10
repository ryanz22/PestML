# Explain LibriMix scripts

## Dataset introduction

[LibriMix: An Open-Source Dataset for Generalizable Speech Separation](https://arxiv.org/pdf/2005.11262.pdf)

[LibriMix code repo](https://github.com/JorisCos/LibriMix)

[wsj0-mix generation scripts, in Python](https://github.com/mpariente/pywsj0-mix)

**NO free** WSJ0 dataset found, UPenn WSJ0 dataset is charged for over $1000

[reference site: Voice Separation with an Unknown Number of Multiple Speakers](https://enk100.github.io/speaker_separation/)

## OpenSLR

[OpenSLR is a site devoted to hosting speech and language resources, such as training corpora for speech recognition, and software related to speech recognition](https://www.openslr.org/)

[LibriSpeech ASR corpus](https://www.openslr.org/12/)

Identifier: SLR12

Summary: Large-scale (1000 hours) corpus of read English speech

Category: Speech

License: CC BY 4.0

Downloads (use a mirror closer to you):
dev-clean.tar.gz [337M]   (development set, "clean" speech )   Mirrors: [US]   [EU]   [CN]  
dev-other.tar.gz [314M]   (development set, "other", more challenging, speech )   Mirrors: [US]   [EU]   [CN]  
test-clean.tar.gz [346M]   (test set, "clean" speech )   Mirrors: [US]   [EU]   [CN]  
test-other.tar.gz [328M]   (test set, "other" speech )   Mirrors: [US]   [EU]   [CN]  
train-clean-100.tar.gz [6.3G]   (training set of 100 hours "clean" speech )   Mirrors: [US]   [EU]   [CN]  
train-clean-360.tar.gz [23G]   (training set of 360 hours "clean" speech )   Mirrors: [US]   [EU]   [CN]  
train-other-500.tar.gz [30G]   (training set of 500 hours "other" speech )   Mirrors: [US]   [EU]   [CN]  
intro-disclaimers.tar.gz [695M]   (extracted LibriVox announcements for some of the speakers )   Mirrors: [US]   [EU]   [CN]  
original-mp3.tar.gz [87G]   (LibriVox mp3 files, from which corpus' audio was extracted )   Mirrors: [US]   [EU]   [CN]  
original-books.tar.gz [297M]   (Project Gutenberg texts, against which the audio in the corpus was aligned )   Mirrors: [US]   [EU]   [CN]  
raw-metadata.tar.gz [33M]   (Some extra meta-data produced during the creation of the corpus )   Mirrors: [US]   [EU]   [CN]  
md5sum.txt [600 bytes]   (MD5 checksums for the archive files )   Mirrors: [US]   [EU]   [CN]  

## LibriMix code study

### generate_librimix.sh

download LibriSpeech dev-clean dataset and unzip to $storage_dir/LibriSpeech

download LibriSpeech test-clean dataset and unzip to $storage_dir/LibriSpeech

download LibriSpeech train-clean-100 dataset and unzip to $storage_dir/LibriSpeech

download LibriSpeech train-clean-360 dataset and unzip to $storage_dir/LibriSpeech

download wham_noise dataset and unzip to $storage_dir

'wait' to make sure all background scripts finish.

call 'augment_train_noise.py' in the first time run.

call 'create_librimix_from_metadata.py' to generate librimix dataset.

### augment_train_noise.py

preprocess wham_noise dataset called by generate_librimix.sh.

'tr' stands for train
'tt' stands for test
'cv' stands for dev also means val

the 'tr' folder should have 20K original noise sound track. The 'augment_train_noise.py' will augment the 20K sound tracks in 'tr' folder by adding 0.8 and 1.2 speed sound tracks which make the total sound track count to 60K.

### create_librimix_from_metadata.py

This Python code is used to generate mixed audio samples from clean speech data and noise data. Here is an explanation:

- It takes as input the paths to the LibriSpeech dataset, WHAM noise dataset, and LibriMix metadata files. It also takes parameters like output directory, sampling rates, number of sources, modes (min/max), and mix types.

- It loops through the LibriMix metadata CSV files. For each file:

  - It creates output directories based on the sampling rates, modes, mix types.
  
  - It reads the metadata - gets the mixture ID, source file paths, gains, noise file path etc.
  
  - It reads the source audio files and noise file.
  
  - It normalizes the source signals and noise by the specified gains.
  
  - It resamples them to the target frequency.
  
  - It pads or truncates the sources to a common length based on the min/max mode.
  
  - It mixes the sources and noise to generate the final mixture audio.
  
  - It computes the SNR values.
  
  - It writes the mixture, sources and noise as WAV files to the output directories.
  
  - It also saves CSV files containing metadata like mixture ID, paths, SNRs etc.

So in summary, it automates the process of generating mixed audio by reading metadata, combining sources and noise, processing, mixing and saving with proper folder structure and metadata. The output is a dataset containing mixtures suitable for training speech separation models.

### create_librimix_metadata.py

This Python code is used to generate metadata for creating mixed audio samples from the LibriSpeech and WHAM datasets. Here is an explanation:

- It takes as input the paths to the LibriSpeech and WHAM datasets and their metadata CSV files. It also takes the number of sources to mix as input.

- It first checks if metadata files already exist for the given inputs to avoid overwriting. 

- It reads the LibriSpeech and WHAM metadata CSVs as pandas DataFrames.

- It generates combinations of utterances from LibriSpeech to mix together. It ensures the utterances are from different speakers.

- For each utterance combination, it picks a random noise sample from WHAM ensuring it is longer than the utterances.

- It reads the audio files for the utterances and noise based on the metadata.

- It normalizes the loudness of audio sources randomly between a given range.

- It mixes the sources and noise together.

- It checks if the mixed signal clips and renormalizes if necessary. 

- It computes the gain values between original and normalized loudness.

- It creates rows for the output metadata DataFrames with info like mixture ID, source paths, gains etc.

- It saves the metadata DataFrames as CSVs containing all the info needed to mix the sources and noise to create the dataset.

So in summary, it automatically generates metadata to create mixtures of LibriSpeech utterances and WHAM noise samples in a configurable way. The output CSVs can be used to actually mix and create the dataset.

### create_librispeech_metadata.py

This Python code defines a command-line utility that generates metadata files corresponding to the downloaded data in the LibriSpeech dataset. The LibriSpeech dataset is a corpus of read speech in English, designed for training and evaluating speech recognition systems. The metadata files contain information about the audio files in the dataset, such as their length, speaker ID, and gender.

The code uses several Python libraries, including os, argparse, soundfile, pandas, glob, and tqdm. Here's an overview of what each library is used for:

os: provides functions for interacting with the operating system, such as creating directories and reading file paths.
argparse: provides a convenient way to parse command-line arguments and options.
soundfile: provides a simple way to read and write sound files in various formats, such as WAV and FLAC.
pandas: provides a powerful data analysis and manipulation library, used here to create and manipulate dataframes.
glob: provides a way to search for files matching a specified pattern.
tqdm: provides a progress bar to track the progress of long-running operations.
The code defines a main function that takes an argument --librispeech_dir specifying the root directory of the LibriSpeech dataset. The function creates a metadata directory within the LibriSpeech directory and then calls the create_librispeech_metadata function to generate metadata files for each subdirectory in the dataset.

The create_librispeech_metadata function, in turn, calls several helper functions to generate metadata for each subdirectory. These functions read various metadata files in the LibriSpeech directory to gather information about the speakers and the audio files, and then create a pandas dataframe containing this information. The dataframe is then filtered to remove audio files shorter than 3 seconds, sorted by length, and saved as a CSV file in the metadata directory.

Overall, this code provides a useful tool for working with the LibriSpeech dataset and generating metadata files that can be used for various tasks, such as training and evaluating speech recognition models.

Sample CSV content:

```csv
speaker_ID,sex,subset,length,origin_path
8975,F,train-clean-100,48000,train-clean-100/8975/270782/8975-270782-0029.flac
200,F,train-clean-100,48000,train-clean-100/200/124140/200-124140-0012.flac
27,M,train-clean-100,48000,train-clean-100/27/124992/27-124992-0034.flac
```

### create_wham_metadata.py

This Python script appears to generate metadata for sound files stored in the WHAM! noise dataset. WHAM! is an open-source dataset used for training and evaluating source separation algorithms, specifically under conditions of additive noise. 

The script is designed to be run from the command line and takes one argument, `wham_dir`, which is the root directory of the WHAM! noise dataset. Here is a high-level breakdown of the script:

1. Import necessary libraries: `os` for interacting with the operating system, `argparse` for parsing command-line arguments, `soundfile` for handling sound files, `pandas` for creating and managing dataframes, `glob` for file path pattern matching, and `tqdm` for progress bars.

2. Define two global variables, `NUMBER_OF_SECONDS` and `RATE`. These are used later in the code to filter out sound files shorter than 3 seconds.

3. Define an `argparse` ArgumentParser to handle command-line arguments. In this case, the only argument is the path to the WHAM! noise dataset directory.

4. Define the `main()` function, which takes the command-line arguments, identifies the metadata directory, and calls the function to create the metadata.

5. Define the `create_wham_noise_metadata()` function. This function iterates over each subdirectory of the WHAM! noise dataset that hasn't been processed yet, generates a metadata dataframe for each subdirectory, filters out sound files shorter than 3 seconds, and writes the metadata to a .csv file.

6. Define the `check_already_generated()` function. This function checks if metadata files have already been generated for each of the subdirectories and returns a list of subdirectories that still need to be processed.

7. Define the `create_wham_noise_dataframe()` function. This function generates a pandas dataframe that gathers information about the sound files in a WHAM! noise subdirectory. It uses the `soundfile` library to find the length of each sound file, and checks if certain strings ('sp08' or 'sp12') are in the file path, which could indicate augmented data. This function returns the created dataframe.

8. Run the `main()` function if the script is run directly from the command line.

The script produces a .csv metadata file for each subset of the WHAM! noise data ('cv', 'tr', 'tt'), where each row corresponds to a sound file and contains columns for the noise ID, the subset it belongs to, the length of the sound file, a boolean indicating whether the file was augmented, and the file's relative path.

## wsjmix code study

This script appears to be used for processing and mixing audio files. Here's a breakdown of what each section does:

Imports: The script uses several libraries such as pathlib, glob, pandas, soundfile, numpy, scipy.signal, tqdm, and argparse.

Arguments: argparse is used to handle command-line arguments. The arguments are wsj0_path, output_folder, n_src, samplerate, and len_mode. wsj0_path specifies the location of the wsj0 dataset, output_folder is the directory where output files will be saved, n_src indicates the number of sources to mix, samplerate is the target sampling rate for the audio files, and len_mode determines how the length of mixed audio files should be treated.

Activity Level: The script reads in activity level files which indicate the volume level of the audio. The code constructs a dictionary mapping utterance ids to their activity levels.

Functions: Two functions, wavwrite_quantize and wavwrite, are defined. wavwrite_quantize quantizes samples for audio processing. wavwrite is used to save the quantized audio samples to a file. The saved audio files have a 16-bit depth.

File Structure: For each condition (tr, cv, tt), the script creates a series of directories for storing the output files.

File Reading: For each condition, the script reads in a file that lists the signal-to-noise ratio (SNR) for each source.

Audio Processing: In the main loop, for each entry in the SNR file, it reads the audio files, resamples them to the target sample rate, pads them to equal length, scales them according to their activity levels and SNR, and then mixes them together.

Output: Finally, the script saves the mixed audio and its component sources. The script can output the shortest, longest, or both versions of the mixed audio, depending on the len_mode argument. For the shortest version, it trims the mixed audio to the length of the shortest source. For the longest version, it pads the mixed audio to the length of the longest source.

Overall, this script is used to mix different sources of audio files together, which could be useful for audio data augmentation or creating data for machine learning tasks such as source separation or speech recognition.
