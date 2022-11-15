import categories
import streamlit as st
from app import swap_app

from utils.predict_new_data import Predictor
from config.help_messages import IMPRESS_TEXT, NO_CONFIG_HELP

CATEGORY = categories.PREDICT
TITLE = "Predict behaviors"

def main(config=None):
    st.markdown("""---""")

    if config is not None:
        st.warning("If you did not do it yet, remove and reupload the config file to make sure that you use the latest configuration!")
        try:
            predictor = Predictor(config)
            predictor.main()

            #show results (optional)

        except FileNotFoundError:
            st.error("Train a classifier first.")
    else:
        st.error(NO_CONFIG_HELP)

    bottom_cont = st.container()
    with bottom_cont:

        st.markdown("""---""")
        st.write('')
        button_col1, button_col2, button_col3, button_col4, button_col5 = st.columns([3, 3, 1, 1, 1])
        if button_col1.button('◀  PRIOR STEP'):
            swap_app('C-auto-active-learning')
        if button_col5.button('NEXT STEP ▶'):
            swap_app('F-view')
        st.write('')
        st.markdown('<span style="color:grey">{}</span>'.format(IMPRESS_TEXT), unsafe_allow_html=True)
