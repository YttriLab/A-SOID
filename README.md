# A-SOiD: An active learning platform for expert-guided, data efficient discovery of behavior.

[![GitHub stars](https://img.shields.io/github/stars/YttriLab/A-SOID.svg?style=social&label=Star)](https://github.com/YttriLab/A-SOID)
[![GitHub forks](https://img.shields.io/github/forks/YttriLab/A-SOID.svg?style=social&label=Fork)](https://github.com/YttriLab/A-SOID)
[![DOI](https://zenodo.org/badge/558906663.svg)](https://zenodo.org/doi/10.5281/zenodo.10210508)

### Read the [paper](https://doi.org/10.1038/s41592-024-02200-1)!
> IF YOU ARE NEW HERE: CHECK OUT THE OVERVIEW AND INSTALLATION SECTION TO GET STARTED!
---
## News:
### February 2024: Paper published and A-SOiD v0.3.1 released!

[Paper](https://doi.org/10.1038/s41592-024-02200-1)

[New Release](https://github.com/YttriLab/A-SOID/releases/tag/v0.3.1)

### August 2023: A-SOiD v0.3.0 is released!
Our first major update is launched! This update includes major changes in the workflow and the GUI.

- **New features**:
  - New manual refinement step: allows users to refine the classification of individual bouts.
  - Predict tab: Ethogram, Pie charts, Stats and  **Videos!**
- **Extended features**:
  - Discover tab: simultaneous split of multiple behaviors in one go
  - Data upload: Sample rate of annotation files can now be set explicitly during upload.
  - Active learning: Confidence threshold now accessible in the GUI
- **Other changes:**
  - Several UX improvements
  - Bug fixes
  - Increased performance during active learning

### How to update:
> Please be aware that this update is not backwards compatible with previous versions of A-SOiD.
1. Download or clone the latest version of A-SOiD from this repository.
2. Create a **new** environment using the `asoid.yml` file (see below).
````
conda env create --file asoid.yml
````
3. Activate the environment you installed A-SOiD in.
4. Start A-SOiD and use it just like before.
````
asoid app
````
---

### Introduction:

[DeepLabCut](https://github.com/AlexEMG/DeepLabCut) <sup>1,2,3</sup>, 
[SLEAP](https://github.com/murthylab/sleap) <sup>4</sup>, and 
[OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) <sup>5</sup> 
have revolutionized the way behavioral scientists analyze data. 
These algorithm utilizes advances in computer vision and deep learning to automatically estimate poses. 
Interpreting the positions of an animal can be useful in studying behavior; 
however, it does not encompass the whole dynamic range of naturalistic behaviors. 

Behavior identification and quantification techniques have undergone rapid development.
To this end, supervised or unsupervised methods (such as [B-SOiD](https://github.com/YttriLab/B-SOID)<sup>6</sup> ) are chosen based upon their intrinsic strengths and weaknesses 
(e.g. user bias, training cost, complexity, action discovery).

Here, a new active learning platform, A-SOiD, blends these strengths and in doing so,
overcomes several of their inherent drawbacks. A-SOiD iteratively learns user-defined
groups with a fraction of the usual training data while attaining expansive classification
through directed unsupervised classification.

## Overview: 

![DLS_Stim](asoid/images/GUI_overview.png)

A-SOiD is a streamlit-based application that integrates the core features of [A-SOiD](https://www.biorxiv.org/content/10.1101/2022.11.04.515138v1) into a user-friendly,
no-coding required GUI solution that can be downloaded and used on custom data.

For this we developed a multi-step pipeline that guides users, independent of their previous machine learning abilities
through the process of generating a well-trained classifier for their own use-case.

In general, users are required to provide a small labeled data set (ground truth) with behavioral categories of
their choice using one of the many available labeling tools (e.g. [BORIS](https://www.boris.unito.it/)<sup>7</sup> ) or
import their previous supervised machine learning data sets. Following the upload of data
(see Fig. above, a), a A-SOiD project is created, including several parameters that further enable
users to select individual animals (in social data) and exclude body parts from the feature extraction.

## Input:

A-SOiD supports the following input types:

### Pose estimation:
- [SLEAP](https://sleap.ai/)
- [DeepLabCut](http://www.mackenziemathislab.org/deeplabcut)

### Annotation files:
- [BORIS](https://www.boris.unito.it/) -> exported as binary files in 0.1 sec time steps (10 Hz): [Read the docs](http://www.boris.unito.it/user_guide/export_events/#export-events-as-behaviors-binary-table)
- any annotation files in this style ([one-hot encoded](https://en.wikipedia.org/wiki/One-hot)), including an index that specifies time steps in seconds.
----
> You can see an example of this using pandas in our docs: [Convert annotations to binary format](docs/export_annotations_to_binary_format.ipynb)
---

## System Requirements
### Hardware requirements
A-SOiD requires only a standard computer with enough RAM to support the model training during active learning. For clustering, CPU and RAM requirements are increased. Refer to [B-SOiD](https://github.com/YttriLab/B-SOID) or our paper for details. 

### Software requirements
#### OS Requirements
This package is supported for *Windows* and *Mac* but can be run on *Linux* computers given additional installation of require packages.

#### Python Dependencies
For dependencies please refer to the requirements.txt file. No additional requirements.

## Installation

> **Note**: This is a quick guide to get you started. For a more detailed guide, see [Installation](docs/installation_stepbystep.md).

### Download repository and install locally as python package within a new environment

Clone this repository and create a new environment in which A-SOiD will be installed automatically (recommended) [Anaconda/Python3](https://www.anaconda.com/).

#### Change your current working directory to the location where you want the directory to be made.
````
cd path/to/A-SOiD
````
Clone it directly from GitHub
```bash
git clone https://github.com/YttriLab/A-SOID.git
```
or download ZIP and unpack where you want. 

### Install A-SOiD using conda (recommended)

1. Create a **new** environment using the `asoid.yml` file (see below).
````
conda env create --file asoid.yml
````

## How to start A-SOiD:

1. Activate the environment you installed A-SOiD in.
````
conda activate asoid
````

2. You can run A-SOiD now from inside your environment:
````
asoid app
````
## Demo:

We invite you to test A-SOiD using the [CalMS21](https://data.caltech.edu/records/s0vdx-0k302) data set. The data set can be used within the app by simply specifying the path to the train and test set files (see below). While you can reproduce our figures using the provided notebooks, the data set also allows an easy first use case to get familiar with all significant steps in A-SOiD.

1. Download the data set and convert it into npy format using their provided script.
2. Run A-SOiD and select 'CalMS21 (paper)' in the `Upload Data` tab.
3. Enter the full path to both train and test files from the first challenge (e.g. 'C:\Dataset\task1_classic_classification\calms21_task1_train.npy' and 'C:\Dataset\task1_classic_classification\calms21_task1_test.npy').
4. Enter a directory and prefix to create the A-SOiD project in.
5. Click on 'Preprocess' and follow the remaining workflow of the app. After successful importing the data set, you can now run the CalMS21 project as any other project in A-SOiD. 

The overall runtime depends on your setup and parameters set during training, but should be completed within 1h of starting the project.
Tested on: AMD Ryzen 9 6900HX 3.30 GHz and 16 GB RAM; Windows 11 Home

---
## Contributors:

A-SOiD was developed as a collaboration between the Yttri Lab and Schwarz Lab by:

[Jens Tillmann](https://github.com/JensBlack), University Bonn

[Alex Hsu](https://github.com/runninghsus), Carnegie Mellon University

[Martin Schwarz](https://github.com/SchwarzNeuroconLab), University Bonn

[Eric Yttri](https://github.com/YttriLab), Carnegie Mellon University

---
## Get in contact:

### Corresponding authors:

Martin K. Schwarz [SchwarzLab](https://ieecr-bonn.de/ieecr-groups/schwarz-group/)

Eric A. Yttri [YttriLab](https://labs.bio.cmu.edu/yttri/)

### Contributing

For recommended changes that you would like to see, open an issue. 

There are many exciting avenues to explore based on this work. 
Please do not hesitate to contact us for collaborations.

### Issues running A-SOiD
If you are having issues, please refer to our issue page first, to see whether a similar issue was already solved.
If this does not apply to your problem, please submit an issue with enough information that we can replicate it. Thank you!

## License
A-SOiD is released under a [Clear BSD License](https://github.com/YttriLab/A-SOID/blob/main/LICENSE) and is intended for research/academic use only.

---

## References
If you are using our work, please make sure to cite us and any additional resources you were using

1. [Mathis A, Mamidanna P, Cury KM, Abe T, Murthy VN, Mathis MW, Bethge M. DeepLabCut: markerless pose estimation of user-defined body parts with deep learning. Nat Neurosci. 2018 Sep;21(9):1281-1289. doi: 10.1038/s41593-018-0209-y. Epub 2018 Aug 20. PubMed PMID: 30127430.](https://www.nature.com/articles/s41593-018-0209-y)

2. [Nath T, Mathis A, Chen AC, Patel A, Bethge M, Mathis MW. Using DeepLabCut for 3D markerless pose estimation across species and behaviors. Nat Protoc. 2019 Jul;14(7):2152-2176. doi: 10.1038/s41596-019-0176-0. Epub 2019 Jun 21. PubMed PMID: 31227823.](https://doi.org/10.1038/s41596-019-0176-0)

3. [Insafutdinov E., Pishchulin L., Andres B., Andriluka M., Schiele B. (2016) DeeperCut: A Deeper, Stronger, and Faster Multi-person Pose Estimation Model. In: Leibe B., Matas J., Sebe N., Welling M. (eds) Computer Vision – ECCV 2016. ECCV 2016. Lecture Notes in Computer Science, vol 9910. Springer, Cham](http://arxiv.org/abs/1605.03170)

4. [Pereira, T.D., Tabris, N., Matsliah, A. et al. SLEAP: A deep learning system for multi-animal pose tracking. Nat Methods 19, 486–495 (2022).](https://doi.org/10.1038/s41592-022-01426-1)

5. [Cao Z, Hidalgo Martinez G, Simon T, Wei SE, Sheikh YA. OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields. IEEE Trans Pattern Anal Mach Intell. 2019 Jul 17. Epub ahead of print. PMID: 31331883.](https://doi.org/10.1109/TPAMI.2019.2929257).
6. [Hsu AI, Yttri EA. B-SOiD, an open-source unsupervised algorithm for identification and fast prediction of behaviors. Nat Commun. 2021 Aug 31;12(1):5188](https://doi.org/10.1038/s41467-021-25420-x)
7. [Friard, O. and Gamba, M. (2016), BORIS: a free, versatile open-source event-logging software for video/audio coding and live observations.  Methods Ecol Evol, 7: 1325-1330](https://doi.org/10.1111/2041-210X.12584)

### How to cite us:

> Tillmann, J.F., Hsu, A.I., Schwarz, M.K. et al. A-SOiD, an active-learning platform for expert-guided, data-efficient discovery of behavior.
> Nat Methods (2024). https://doi.org/10.1038/s41592-024-02200-1

or see [Cite Us](CITATION)



