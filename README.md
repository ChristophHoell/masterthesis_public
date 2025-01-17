# Text Guided Generation of Head Avatars

## Project Structure:

1. ``Proposed Method`` Final Version of the Thesis implementation
2. ``MDM Adapted`` Implementation of the Human Motion Diffusion Model adapted to the facial domain
3. ``SMF Adapted`` Implementation of the Stable MoFusion Project adapted to the facial domain

## Environment Setup:

Install the conda environment with the following command:
```
conda env create --name thesis --file requirements.yml
conda activate thesis
```
In case the CLIP language model is not installed correctly through conda, install it with the following command:
```
python -m pip install git+https://github.com/openai/CLIP.git
```

## Proposed Method:
Our Proposed method is a Transformer cVQVAE with the goal of generating a 3D facial animation solely from a textual description.

![A person is kissing.](https://github.com/ChristophHoell/masterthesis_public/blob/main/assets/kissing.gif)
"A person is kissing."
![A person is sneering.](https://github.com/ChristophHoell/masterthesis_public/blob/main/assets/sneering.gif)
"A person is sneering."

We demonstrate that our method is capable of generating realistic facial motions from a textual description while maintaining good sequential coherence.


![A person is yawning.](https://github.com/ChristophHoell/masterthesis_public/blob/main/assets/yawning_diversity.gif)
"A person is yawning."

We also demonstrate the non-deterministic behavior of our model with regards to the diversity in output for the same prompt.

For more information we refer to the written [**Thesis**](https://github.com/ChristophHoell/masterthesis_public/blob/main/assets/Thesis.pdf) and the [**Presentation**](https://github.com/ChristophHoell/masterthesis_public/blob/main/assets/Presentation.pdf).