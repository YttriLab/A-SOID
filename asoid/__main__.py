import click
import os
import configparser as cfg

import sys
# get current dir
dirname = os.path.dirname(__file__)
sys.path.append(dirname)

from headless import Project

def load_streamlit_config(dirname):
    config = cfg.ConfigParser()
    #make config case sensitive
    config.optionxform = str
    config.read(os.path.join(dirname, "config/streamlit_config.ini"))
    #convert config to dict
    config = {s: dict(config.items(s)) for s in config.sections()}
    return config

def conv_config_to_args(config):
    #translate config to streamlit args
    args = []
    for k, v in config.items():
        for kk, vv in v.items():
            args.append('--' + k + '.' + kk)
            args.append(vv)
    return args

@click.group()
def main():
    pass

@main.command("app")
def main_streamlit():
    """Runs the A-SOiD streamlit app"""
    #get current dir
    dirname = os.path.dirname(__file__)

    #load config
    config = load_streamlit_config(dirname)
    #convert config to dict
    args = conv_config_to_args(config)

    filename = os.path.join(dirname, 'app.py')
    # run streamlit app using the hideous way and pass args
    # because streamlit has no native support for this
    os.system('streamlit run ' + filename + ' ' + ' '.join(args))

@main.command("predict")
@click.argument('project_path', type = click.Path(exists=True)
                , required=True)
@click.argument('pose_files', type = list, required=True)
@click.option("-origin", "pose_origin", type = str, default = None
              , help="origin of the pose files, if not specified, the origin from the config file is used")
@click.option("-verbose", "verbose", is_flag=True, default=False, help="verbose output")
def main_predict(project_path, pose_files, pose_origin, verbose):
    """Uses A-SOiD classifier to predict on new pose files.

    :param project_path: path to project containing config file

    :param pose_files: list of paths to pose files, or path to folder with pose files, or path to single pose file

    :param pose_origin: origin of the pose files, if not specified, the origin from the config file is used

    :param verbose: verbose output

    :output: the predictions are saved as .csv files in the same directory as the pose files"""
    click.echo("Predicting on new pose files...")
    asoid_project = Project(project_path, verbose=verbose)
    _ = asoid_project.predict(pose_files, pose_origin)


if __name__ == "__main__":
    main()