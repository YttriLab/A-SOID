import categories
import streamlit as st
from app import swap_app
from config.help_messages import IMPRESS_TEXT, NO_CONFIG_HELP
from utils.view_results import Viewer
from utils.motionenergy import MotionEnergyMachine

CATEGORY = categories.VIEW
TITLE = "View"

def main(config=None):
    st.markdown("""---""")
    viewer_tab, motion_energy_tab = st.tabs(["Viewer", "Motion Energy"])
    with viewer_tab:
        st.header("View annotation files or predictions")
        viewer = Viewer(config)
        viewer.main()
    with motion_energy_tab:

        st.header("Create animations and calculate motion energy")
        if config is not None:
            motion_energy = MotionEnergyMachine(config)
            motion_energy.main()

        else:
            st.warning(NO_CONFIG_HELP)



    bottom_cont = st.container()
    with bottom_cont:

        st.markdown("""---""")
        st.write('')
        button_col1, button_col2, button_col3, button_col4, button_col5 = st.columns([3, 3, 1, 1, 1])
        if button_col1.button('◀  PRIOR STEP'):
            swap_app('E-predict')
        if button_col5.button('NEXT STEP ▶'):
            swap_app('G-unsupervised-discovery')
        st.write('')
        st.markdown('<span style="color:grey">{}</span>'.format(IMPRESS_TEXT), unsafe_allow_html=True)
