from setuptools import setup, find_packages
from os import path
import asoid


curr_dir = path.abspath(path.dirname(__file__))

with open("README.md", "r") as fh:
    long_description = fh.read()

def get_requirements():
    with open(path.join(curr_dir, "requirements.txt"), encoding="utf-8") as f:
        return f.read().strip().split("\n")

setup(
    name='asoid',
    version= asoid.__version__,
    setup_requires=["setuptools_scm"],
    install_requires=get_requirements(),
    description='ASOiD: An active learning approach to behavioral classification',
    long_description=long_description,
    long_description_content_type="text/markdown",
    project_urls={
        "Documentation": "https://github.com/YttriLab/A-SOID",
        "Bug Tracker": "https://github.com/YttriLab/A-SOID/issues"
    },
    url="https://github.com/YttriLab/A-SOID",
    author=['Jens F. Schweihoff','Alexander Hsu'],
    keywords="Active learning, pose estimation, behavior analysis, neuroscience",
    license="The Clear BSD License",
    packages=find_packages(),  # same as name
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "asoid =asoid.__main__:main"
        ]
    },
    python_requires=">=3.7",
)
