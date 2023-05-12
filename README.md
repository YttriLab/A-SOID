# A-SOiD: An active learning platform for expert-guided, data efficient discovery of behavior.

[![GitHub stars](https://img.shields.io/github/stars/YttriLab/A-SOID.svg?style=social&label=Star)](https://github.com/YttriLab/A-SOID)
[![GitHub forks](https://img.shields.io/github/forks/YttriLab/A-SOID.svg?style=social&label=Fork)](https://github.com/YttriLab/A-SOID)

### Read the [preprint](https://www.biorxiv.org/content/10.1101/2022.11.04.515138v1)!

[DeepLabCut](https://github.com/AlexEMG/DeepLabCut) <sup>1,2,3</sup>, 
[SLEAP](https://github.com/murthylab/sleap) <sup>4</sup>, and 
[OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) <sup>5</sup> 
have revolutionized the way behavioral scientists analyze data. 
These algorithm utilizes recent advances in computer vision and deep learning to automatically estimate 3D-poses. 
Interpreting the positions of an animal can be useful in studying behavior; 
however, it does not encompass the whole dynamic range of naturalistic behaviors. 

Behavior identification and quantification techniques have undergone rapid development.
To this end, supervised or unsupervised methods (such as [B-SOiD](https://github.com/YttriLab/B-SOID)<sup>6</sup> ) are chosen based upon their intrinsic strengths and weaknesses 
(e.g. user bias, training cost, complexity, action discovery).

Here, a new active learning platform, A-SOiD, blends these strengths and in doing so,
overcomes several of their inherent drawbacks. A-SOiD iteratively learns user-defined
groups with a fraction of the usual training data while attaining expansive classification
through directed unsupervised classification.

To facilitate use, A-SOiD comes as an intuitive, open-source interface for efficient segmentation
of user-defined behaviors and discovered subactions.

## Overview: 

![GUI overview](asoid/images/GUI_overview.png)

A-SOiD is a streamlit-based application that integrates the core features of [A-SOiD](https://www.biorxiv.org/content/10.1101/2022.11.04.515138v1) into a user-friendly,
no-coding required GUI solution that can be downloaded and used on custom data.


For this we developed a multi-step pipeline that guides users, independent of their previous machine learning abilities
through the process of generating a well-trained, semi-supervised classifier for their own use-case.

In general, users are required to provide a small labeled data set (ground truth) with behavioral categories of
their choice using one of the many available labeling tools (e.g. [BORIS](https://www.boris.unito.it/)<sup>7</sup> ) or
import their previous supervised machine learning data sets. Following the upload of data
(see Fig. above, a), a A-SOiD project is created, including several parameters that further enable
users to select individual animals (in social data) and exclude body parts from the feature extraction.


Based on the configuration, the feature extraction (see Fig. below, b top) can be further customized
by defining a "bout length" referring to the temporal resolution in which single motifs are expected to appear
(e.g. the shortest duration a definable component of the designated behavior is expected to last).
The extracted features are then used in combination with the labeled ground truth to train a baseline model.
Here, an initial evaluation will give users insight into the performance on their base data set (see Fig. above, b bottom).
Note, that different splits are used to allow for a more thorough analysis (see Publication Methods for further details).


The baseline classification will then be used as a basis for the first active learning iteration,
where users are prompted by the app to view and refine bouts that were classified with low confidence
by the baseline model (see Fig. above,c left). Bouts are visualized by showing an animated sequence
of the provided pose information and designated body parts and the viewer can be utilized to show the bouts
in several options, including increased/decreased speed, reverse view and frame-by-frame view.
After submission of a refined bout, a new bout is shown at its place and the refinement continues for
a user-defined amount of low confidence bouts. Following refinement, a new iteration of the model is trained
and its performance can be viewed (see Fig. above,c right) in comparison to previous iterations.
This process is then repeated until the user is satisfied with the model's performance or until a plateau
has been reached (see publication).
\newline

Finally, users can upload and classify new data using the app and the previously trained classifier
(see Fig. above,d). To gain further insight into the results of the classification,
the app offers a reporting tab that allows users to view results (see Fig. above,d).

## Input:

A-SOiD supports the following input types:

### Pose estimation:
- [SLEAP](https://sleap.ai/)
- [DeepLabCut](http://www.mackenziemathislab.org/deeplabcut)

### Annotation files:
- [BORIS](https://www.boris.unito.it/) -> exported as binary files in 0.1 sec time steps (10 Hz): [Read the docs](https://boris.readthedocs.io/en/latest/#export-events-as-behavioral-binary-table)
- any annotation files in this style ([one-hot encoded](https://en.wikipedia.org/wiki/One-hot)), including an index that specifies time steps in seconds.
----
> You can see an example of this using pandas in our docs: [Convert annotations to binary format](docs/export_annotations_to_binary_format.ipynb)
---

## System Requirements
### Hardware requirements
A-SOiD requires only a standard computer with enough RAM to support the model training during active learning. For clustering, CPU and RAM requirements are increased. Refer to [B-SOiD](https://github.com/YttriLab/B-SOID) or our paper for details. 

### Software requirements
#### OS Requirements
This package is supported for *Windows* but can be run on *Linux* computers given additional installation of require packages.

#### Python Dependencies
For dependencies please refer to the requirements.txt file. No additional requirements.

## Installation

There are two ways to install A-SOiD. Installation will only take a couple of minutes. We recommend using a fresh environment in any case to avoid any installation conflicts.
To simplify the process, we provide `asoid.yml` file that will do everything for you (see below).

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

#### Create an environment using anaconda:
````
conda env create --file asoid.yml
````

#### Alternatively, you can install A-SOiD locally 
in the directory you saved the repo in:
````
cd path/to/A-SOiD
````
activate the environment, you want to install A-SOiD in:
````
conda activate MYENVIRONMENT
````
install using `setup.py` in your own environment:
````
pip install .
````

A-SOiD is installed alongside all dependencies.

## Updating A-SOiD

1. Download or clone the latest version of A-SOiD from this repository.

2. Activate the environment you installed A-SOiD in.
````
conda activate asoid
````
3. Go to the locotion of that you unpacked the latest version at.
````
cd path/to/A-SOiD
````
4. Install the new version on-top of the other using `setup.py`:
````
pip install .
````
The console output should look like this:

    Successfully built asoid
    Installing collected packages: asoid
      Attempting uninstall: asoid
        Found existing installation: asoid 0.1
        Uninstalling asoid-0.1:
          Successfully uninstalled asoid-0.1
    Successfully installed asoid-0.2.0

You can start A-SOiD again and use the new version just like before.

## How to start A-SOiD:

````
conda activate asoid
````

You can run A-SOiD now from inside your environment by using (you do not have change directories anymore):
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

# Running A-SOiD without the GUI
### or: How to use asoid.core and asoid CLI


There are two ways on how to interact with asoid without the GUI.


1. asoid CLI, which is a command line interface to asoid.core. The CLI is a wrapper around asoid.core and can be used to run some external functionalities of asoid. The CLI is also used to run the GUI itself.
2. asoid.core module, which gives you access to some core functionalities of asoid, such as the prediction of new files using previously trained classifiers.

## asoid CLI

Run the CLI with the following command to get some basic overview of the available commands:
```bash
asoid --help
```
at time of writing the output is:
```bash
Usage: asoid [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  app      Runs the A-SOiD streamlit app
  info     Runs the A-SOiD diagnostics
  predict  Uses A-SOiD classifier to predict on new pose files.
```

### Predict new data using a previously trained classifier

```bash
asoid predict [OPTIONS] PROJECT_PATH POSE_FILES
```

```
  Uses A-SOiD classifier to predict on new pose files.

  :param project_path: path to project containing config file

  :param pose_files: list of paths to pose files, or path to folder with pose
  files, or path to single pose file

  :param pose_origin: origin of the pose files, if not specified, the origin
  from the config file is used

  :param verbose: verbose output

  :output: the predictions are saved as .csv files in the same directory as
  the pose files

Options:
  -origin TEXT  origin of the pose files, if not specified, the origin from
                the config file is used
  -verbose      verbose output
  --help        Show this message and exit.
```

### Get some info

The info command is used to run the diagnostics of asoid. Use this function if your are reporting an issue, to give us some information about your system and the asoid installation.

```bash
asoid info
```
The output might look like this:
```bash
A-SOiD diagnostics:
Python version: 3.8.15 (default, Nov 24 2022, 14:38:14) [MSC v.1916 64 bit (AMD64)]
Python path: C:\Users\Asoidonaut\anaconda3\envs\asoid\python.exe
A-SOiD path: C:\Users\Asoidonaut\anaconda3\envs\asoid\lib\site-packages\asoid
A-SOiD version: 0.3
```

## asoid.core

asoid.core is a python module that gives you access to some core functionalities of asoid. You can use it to run some of the functionalities of asoid without the GUI.

For example, you can use it to predict new data using a previously trained classifier. 

Quick example:
```python
import asoid.core as acore

# load the project
project = acore.Project('path/to/project')
#predict new data
predictions = project.predict('path/to/pose/files')
```

See [Run A-SOiD as a script](docs/run_asoid_as_script.ipynb) for more details.



---
## Contributors:

A-SOiD was developed as a collaboration between the Yttri Lab and Schwarz Lab by:

[Jens Schweihoff](https://github.com/JensBlack), University Bonn

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
    A-SOiD, an active learning platform for expert-guided, data efficient discovery of behavior.
    Jens F. Schweihoff, Alexander I. Hsu, Martin K. Schwarz, Eric A. Yttri
    bioRxiv 2022.11.04.515138; doi: https://doi.org/10.1101/2022.11.04.515138

or see [Cite Us](CITATION)



