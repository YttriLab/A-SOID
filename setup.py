from setuptools import setup, find_packages
from os import path
curr_dir = path.abspath(path.dirname(__file__))

with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
    name='asoid',
    version='0.2.0',
    description='ASOiD: An active learning approach to behavioral classification',
    long_description=long_description,
    long_description_content_type="text/markdown",
    project_urls={
        "Documentation": "https://github.com/YttriLab/A-SOID",
        "Bug Tracker": "https://github.com/YttriLab/A-SOID/issues"
    },
    url="https://github.com/YttriLab/A-SOID",
    author=['Jens F. Schweihoff','Alexander Hsu'],
    entry_points={
        "console_scripts": [
                            "asoid =asoid.__main__:main"
                            ]
        },
    packages=find_packages(),  # same as name
    include_package_data=True,
    install_requires=["matplotlib"
                    ,"numpy"
                    ,"pandas"
                    ,"seaborn"
                      #streamlit 1.12.0 has an error with streamlit.cli...
                    ,"streamlit~=1.11.0"
                    ,"streamlit_option_menu"
                    ,"hydralit"
                    ,"opencv-python"
                    ,"tqdm"
                    ,"stqdm"
                    ,"scikit-learn"
                    ,"h5py"
                    ,"plotly"
                    ,"pillow"
                    ,"joblib"
                    ,"scipy"
                    ,"ipython"
                    ,"psutil"
                    ,"numba"
                    ,"hdbscan"
                    ,"setuptools"
                    ,"umap-learn"
                    ,"click"
                    ,"moviepy"
                    ],
    python_requires=">=3.7",
)
