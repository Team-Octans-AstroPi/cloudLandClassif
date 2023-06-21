# CloudLandClassif
This folder includes the code used to train 2 models used in our ML-based climate classificaton experiment, which are:
- <b>The Clouds ML model</b>, used to identify cloud types, using clouds extracted by [imageSeparationCV2](https://github.com/Team-Octans-AstroPi/imageSeparationCV2)
- <b>The Land Type ML model</b>, used to identify "land types" (ocean, shore, land, land with ice and snow, desert, night, etc.)

Both of these models are MobileNetV1-based, retrained on a Raspberry Pi with a Coral Edge TPU.

## Model Performance
Our cloud classification model achieved 64% test accuracy and 70% on the whole dataset, while the land model achieved 67% test accuracy and 84% on the whole dataset.
