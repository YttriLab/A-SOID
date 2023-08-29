import streamlit as st
from config.help_messages import *
from utils.motionenergy import MotionEnergyMachine


TITLE = "View"

def main(config=None):
    st.markdown("""---""")
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
