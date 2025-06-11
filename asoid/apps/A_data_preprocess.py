import streamlit as st
from asoid.utils.load_preprocess import Preprocess
from asoid.config.help_messages import *


TITLE = "Preprocess data"
PREPROCESS_HELP = ("In this step, you will preprocess the data before extracting features. "
                   "\n\n The data will be used to train the classifier and predict the behavior in the next steps."
                   "\n\n---\n\n"
                   "**Step 1**: Create a new project or upload an existing project configuration."
                   
                   "\n\n **Step 2**: Select a pose estimation origin."
                   "\n\n **Step 3**: Upload pose files."
                     "\n\n **Step 4**: Upload annotation files."
                     "\n\n **Step 5**: Match pose and annotation files in the same order."
                        "\n\n **Step 6**: Set the config parameters."
                     "\n\n **Step 7**: Create the project and preprocess the data."
                   "\n\n **Step 8**: Continue with :orange[Extract Features]."
                   "\n\n---\n\n"
                   ":red[Using the same prefix and working directory will result in an overwrite and might cause unintended problems.]"
                   )

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

def main(config=None):
    st.markdown("""---""")

    if config:
        st.title("View Project Configuration")
        view_config_md(config)
    else:
        st.title("Create project and upload data")
        st.expander("What is this?", expanded=False).markdown(PREPROCESS_HELP)
        processor = Preprocess()
        processor.main()
    st.session_state['page'] = 'Step 2'
    bottom_cont = st.container()
    with bottom_cont:

        st.markdown("""---""")
        st.write('')
        st.markdown('<span style="color:grey">{}</span>'.format(IMPRESS_TEXT), unsafe_allow_html=True)


if __name__ == "__main__":
    main()

