import streamlit as st
import numpy as np
import os
from config.help_messages import IMPRESS_TEXT, NO_CONFIG_HELP
from utils.load_workspace import load_refinement

TITLE = "Create new dataset"


def main(ri=None, config=None):
    st.markdown("""---""")

    if config is not None:
        st.warning(
            "If you did not do it yet, remove and reupload the config file to make sure that you use the latest configuration!")

        working_dir = config["Project"].get("PROJECT_PATH")
        prefix = config["Project"].get("PROJECT_NAME")
        iteration = config["Processing"].getint("ITERATION")
        selected_iter = ri.selectbox('select iteration number', np.arange(iteration+1))
        project_dir = os.path.join(working_dir, prefix)
        iter_folder = str.join('', ('iteration-', str(selected_iter)))
        os.makedirs(os.path.join(project_dir, iter_folder), exist_ok=True)

        if 'refinements' not in st.session_state:
            [video_name,
             st.session_state['scaled_features'],
             st.session_state['predict'],
             st.session_state['examples_idx'], st.session_state['refinements']] = load_refinement(
                working_dir, prefix)
        st.write(st.session_state['scaled_features'].shape)
        st.write(st.session_state['predict'].shape)
        st.write(st.session_state['examples_idx'].keys())
        # st.write(st.session_state['predict'][st.session_state['examples_idx']['other']])
        st.dataframe(st.session_state['examples_idx'])

        create_button = st.button(f'Create iteration {iteration+1} training dataset'.upper())
        # if create_button:




    else:
        st.error(NO_CONFIG_HELP)

    bottom_cont = st.container()
    with bottom_cont:

        st.markdown("""---""")
        st.write('')
        st.markdown('<span style="color:grey">{}</span>'.format(IMPRESS_TEXT), unsafe_allow_html=True)
