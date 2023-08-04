import streamlit as st
from config.help_messages import IMPRESS_TEXT, NO_CONFIG_HELP
from utils.predict_new_data import Predictor

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
        st.markdown('<span style="color:grey">{}</span>'.format(IMPRESS_TEXT), unsafe_allow_html=True)
