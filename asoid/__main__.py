import click
import os
import configparser as cfg

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

if __name__ == "__main__":
    main()