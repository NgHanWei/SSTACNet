# Self-Selecting Semi-Supervised Transformer-Attention Convolutional Network for Four Class EEG-Based Motor Imagery Decoding

### Framework

<img src=".SS-Learn.PNG" alt="framework" width="600"/>

### Results Overview

| Methodology | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | Average |
|-|-|-|-|-|-|-|-|-|-|-|
| ATCNet | 85.92 |  64.26 |  92.42 |  80.51  | 72.20  | 69.31 |  94.95  | 82.67 | 88.09 | 81.15 |
| SSTACNet (Self-Selecting Augmentation only) | 90.25  | 59.21 |  95.31 |  80.87  | 77.62 |  71.12  | 91.34 |  84.48  | 86.64 | 81.87 |
| SSTACNet (Few-Shot Adaptation only) | 84.48  | 63.18  | 92.06 |  80.87 |  69.31 |  70.40 |  93.50 |  83.39  | 89.53 | 80.75 |
| SSTACNet | 89.17  | 75.09  | 94.22  | 88.09  | 80.14  | 73.65 |  91.34  | 85.92  | 92.06 | 85.52 |


## Dependencies

Models were trained and tested by a single GPU, using Python 3.7 with TensorFlow framework. The following packages are required:

TensorFlow 2.7 <br>
matplotlib 3.5 <br>
NumPy 1.20 <br> 
scikit-learn 1.0 <br>
SciPy 1.7

## Datasets

The [BCI Competition IV-2a dataset](https://www.bbci.de/competition/iv/#dataset2a) needs to be downloaded. The dataset can be downloaded from [here](http://bnci-horizon-2020.eu/database/data-sets).

### Run

The dataset path will be used to run the code as follows.

```
python main_TrainValTest.py --path $data_path$
```
### References


