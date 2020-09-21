# Readme

This repository contains code for translation of sign language videos to sign language glosses and spoken language sentences.

## Requirements

- [SLR](https://github.com/imRishabhGupta/Indian-Sign-Language-Recognition) dataset, [PHOENIX14T](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/) dataset or [How2Sign](https://arxiv.org/abs/2008.08143) dataset
- Python (3.6.9), Pytorch (1.5.0), pytorch-lightning (0.7.3), [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) (1.5.1)

## Usage

Run OpenPose on datasets and process obtained files with scripts described in [Keypoint utils](https://github.com/Asdf11x/stt#keypoint-utils) and textual data with scripts described in [Text utils](https://github.com/Asdf11x/stt#text-utils) located in [utils](https://github.com/Asdf11x/stt/tree/master/utils) directory. Run training with scripts described in [Run predictions](https://github.com/Asdf11x/stt#run-prediction) located in [run](https://github.com/Asdf11x/stt/tree/master/run) directory.

## Keypoint utils

- keypoint_visualization.py
	- visualize outputs of OpenPose
	- param1: set path to directory (dir_0) above utterance folder of json files (possible to process mutliple utterance folders at once)
		- dir_0 -> utterance_ID -> kp00.json, kp01.json
- save_files.py
	- save json files to one numpy file
	- param1: set path to json file directory
	- param1: set path to target directory
- keypoint_normalization.py
	- normalize output of OpenPose
	- param1: set path to numpy file
	- param2: set path to target directory

## Text utils

- vocab_utils.py
	- create vocab list for a text file and transform 
    - param1: set path to file containing sentences
    - param2: to transform sentences set to 1, else 0
    - param3: set path to target directory
- merge\_vocab_lists.py
    - merge vocab lists in one folder
    - param1: set path to folder of vocab lists
    - param2: set path to target directory
- npy2sentences\_utils.py and npy2categories\_utils.py
    - create link from the npy files to the sentences files
    - param1: set path to file containing sentences
    - param2: set path to transformed sentences file
    - param3: set path to target folder

## Run prediction

- main.py
    - run main application
    - param1: hparams.json

- hparam.json
    - input_size: 42 for slr dataset, else 256
    - output_size: length of vocab_merged.txt file
    - to test model: set test paths to validation paths