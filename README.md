# A-SOiD: An active learning platform for expert-guided, data efficient discovery of behavior.

[![GitHub stars](https://img.shields.io/github/stars/YttriLab/A-SOID.svg?style=social&label=Star)](https://github.com/YttriLab/A-SOID)
[![GitHub forks](https://img.shields.io/github/forks/YttriLab/A-SOID.svg?style=social&label=Fork)](https://github.com/YttriLab/A-SOID)

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

![DLS_Stim](asoid/images/GUI_overview.png)

A-SOiD is a streamlit-based application that integrates the core features of [A-SOiD](REF_PAPER) into a user-friendly,
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

Pose estimation:
- [SLEAP](https://sleap.ai/)
- [DeepLabCut](http://www.mackenziemathislab.org/deeplabcut)

Annotation files:
- [BORIS](https://www.boris.unito.it/) -> exported as binary files
- any annotation files in this style ([one-hot encoded](https://en.wikipedia.org/wiki/One-hot)), including an indexed that specifies time steps in seconds.
----
_Note_:

You can easily convert annotation files from a categorical encoded type to this format by using functions such as:
````
pandas.get_dummies()
````
or 
````
sklearn.preprocessing.OneHotEncoder
````
---
## Installation

There are three ways to install and run A-SOiD depending on your preferences. All installation methods will result in the same app running.
While some are easier to access, others are easier to alter.

### Download and install locally as app (Easiest)
The easiest solution is to download our exe file and install A-SOiD locally on your computer.

**COMING SOON**

### Download repository and install locally as python package (advanced)

Clone this repository and create a new environment (recommended) or active the environment using [Anaconda/Python3](https://www.anaconda.com/), you want to install A-SOiD in.
Next, install the A-SOiD package by moving to the directory containing A-SOiD:

Git clone the web URL (example below) or download ZIP. 

Change your current working directory to the location where you want the cloned directory to be made.
```bash
git clone https://github.com/YttriLab/A-SOID.git
```

````
cd path/to/A-SOiD
````
and install the package once using the `setup.py`:
````
pip install .
````
You can run A-SOiD now from inside your environment by using (you do not have change directories anymore):
````
asoid app
````

### Download repository and run locally as streamlit app (developer option)

Clone this repository and create a new environment using [Anaconda/Python3](https://www.anaconda.com/) with the provided `environment.yaml` file.

Git clone the web URL (example below) or download ZIP. 

Change your current working directory to the location where you want the cloned directory to be made.
```bash
git clone https://github.com/YttriLab/A-SOID.git
```

````
cd path/to/A-SOiD
````
and create an environment using anaconda:
````
conda env create --file asoid.yaml
````
You can run A-SOiD now from inside your environment by using (you have to change directories to the cloned repo):
````
conda activate asoid

streamlit run asoid/app.py
````
---
## Get in contact:

### Corresponding authors:

Martin K. Schwarz [SchwarzLab](https://ieecr-bonn.de/ieecr-groups/schwarz-group/)

Eric A. Yttri [YttriLab](https://labs.bio.cmu.edu/yttri/)

### Contributing

Pull requests are welcome. For recommended changes that you would like to see, open an issue. 

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





