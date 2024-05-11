# BirdClef 2023 Species Classification

## Description
This project develops a machine learning model to classify species in the BirdClef 2023 dataset. It leverages the Whisper encoder for robust feature extraction and applies convolutional neural network (CNN) variations to accurately determine bird species from audio samples.

## Installation

Before running the project, ensure that you have all the necessary libraries installed. You can install the dependencies using the following command:

```bash
pip install -r requirements.txt
```

## Dataset Structure

Download the dataset used in this project from the [BirdClef 2023 competition on Kaggle](https://www.kaggle.com/competitions/birdclef-2023/data).

The dataset is organized within the `birdclef-2023` directory, structured as follows:

```
birdclef-2023
├── augmented_audio    # Augmented audio files, can be made using BirdClef_Augmentation.ipynb
├── augmented_pt       # Augmented PyTorch tensor files, can be made using needToMakePT = True in BirdClef_Classification.ipynb
├── checkpoints        # Model checkpoints, automatically created to save checkpoints during each epoch
├── train_audio        # Original training audio files, taken from BirdClef 2023 dataset
├── train_pt           # PyTorch tensor files from training audio, can be made using needToMakePT = True in BirdClef_Classification.ipynb
├── aug_metadata.csv   # Metadata for augmented audio files, can be made using BirdClef_Augmentation.ipynb
└── train_metadata.csv # Metadata for training audio files, taken from BirdClef 2023 dataset
BirdClef_Augmentation.ipynb
BirdClef_Classification.ipynb
PrintCheckpoint.ipynb
requirements.txt
...
...
```

Make sure the data is arranged as shown above to properly run the model training and evaluation scripts.

## Configuration

Training and evaluation configurations can be adjusted through the following flags:

- `augmentedRun`: `True` for using augmented data, `False` for raw data.
- `FTRun`: `True` for feature tuning enabled, `False` for it disabled.

These flags determine the saving path of model checkpoints within the `checkpoints` directory.

## Checkpoint

Checkpoints are automatically saved in designated subdirectories within the `checkpoints` folder, corresponding to the configuration of your training session based on the dataset and feature tuning flags.
The demo pretrained model can be accessed here: https://drive.google.com/file/d/1BdgFq3qonHxFxJRxbqSrSMbgfsvj5umb/view?usp=drive_link


## Contributing

We welcome contributions to improve the model and its implementation. If you have suggestions or improvements, please open an issue to discuss your ideas before submitting a pull request.

## Contact

For any queries regarding this project, please open an issue in the GitHub repository, and we will get back to you.
