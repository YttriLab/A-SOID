"""Project creation"""

import os
import configparser as cfg
import streamlit as st


def load_default_config():
    """Loads default config as dictionary from default_config.ini
    :return default config as dict"""
    default_config = cfg.ConfigParser()
    # set so that it imports also Capital and UPPER CASE
    default_config.optionxform = str
    config_path = os.path.join(
        os.path.dirname(__file__), "../config/default_config.ini"
    )
    try:
        with open(config_path) as file:
            default_config.read_file(file)
    except FileNotFoundError:
        raise FileNotFoundError(
            "The default_config.ini was not found. Make sure it exists."
        )
    return default_config


def load_config(project_path: str):
    """Loads project config as dictionary from config.ini
    :param project_path: str. full path to project folder (excluding "config.ini")
    :return project config and config path"""
    project_config = cfg.ConfigParser()
    # set so that it imports also Capital and UPPER CASE
    project_config.optionxform = str

    config_path = project_path + "/config.ini"
    try:
        with open(config_path) as file:
            project_config.read_file(file)
    except FileNotFoundError:
        raise FileNotFoundError("Config file does not exist at this location.")
    return project_config, config_path


def write_ini(config, path: str):
    filename = os.path.basename(path)
    with open(path, "w") as file:
        config.write(file)


def update_config(project_path, updated_params: dict):
    """
    Updates project config.ini with new parameter values given by updated_params dictionary
    :param project_path: path to project
    :param updated_params: dictionary in style dict(section: dict(parameter: value, ...), ...)
    :return: updated config
    """

    config, project_name = load_config(project_path)

    for section, parameters in updated_params.items():
        for parameter, value in parameters.items():
            if isinstance(value, list) or isinstance(value, tuple):
                config.set(section, parameter, ", ".join(map(str, value)))
            else:
                config.set(section, parameter, str(value))

    write_ini(config, project_path + "/config.ini")
    print("Config updated.".format(project_name))

    return config


def create_new_project(base_path: str, project_name: str = None, overwrite=False):
    """Creates folder structure for new project
    General project structe:
    "ProjectName_YYYY/MM/DD"; "input", "iterations", "results", "converted_files"
    + config.ini
    :param base_path: str. Full path to folder where project structure is created
    :param project_name: str. Name of project
    :param overwrite: bool. If project should be overwritten. Default False.
    :return project_path: path where project was created, project_name: name of project
    """

    # folder_dict = dict(
    #     input_folder="input",
    #     conv_folder="converted_files",
    #     model_folder="iterations",
    # )

    base_path = os.path.abspath(base_path)
    config_name = "config"
    # project_name = "{}_{}".format(project_name, date.today().isoformat())
    project_folder = base_path + f"/{project_name}/"
    try:
        os.makedirs(project_folder, exist_ok=overwrite)
    except FileExistsError as e:
        raise FileExistsError("Project already exists!") from e

    # for folder_name in folder_dict.values():
    #     os.makedirs(project_folder + folder_name, exist_ok=True)

    config = load_default_config()
    config.set("Project", "PROJECT_NAME", project_name)
    config.set("Project", "PROJECT_PATH", project_folder)

    write_ini(config, project_folder + "/config.ini")
    print("New project {} created.".format(project_name))

    return project_folder, project_name


def view_config_str(config):
    """Returns content of config file in str format"""
    sections = [x for x in config.keys() if x != "DEFAULT"]
    # remove the "DEFAULT" KEY that holds no info
    for num, section in enumerate(sections):
        print(f"\n{section}: \n")
        for parameter, value in config[sections[num]].items():
            print(f"{parameter.replace('_', ' ').capitalize()}: {value}")

def view_config_md(config):
    """Returns content of config file in markdown format"""
    sections = [x for x in config.keys() if x != "DEFAULT"]
    # remove the "DEFAULT" KEY that holds no info
    clms = st.columns(len(sections))
    for num, section in enumerate(sections):
        with clms[num]:
            placeholder = st.container()
            placeholder_exp = placeholder.expander(f'{section}'.upper(), expanded=True)
            with placeholder_exp:
                # st.subheader(section)
                for parameter, value in config[sections[num]].items():
                    st.markdown(f"{parameter.replace('_', ' ').upper()}: {value}")