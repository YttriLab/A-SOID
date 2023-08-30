import streamlit as st
from utils.project_utils import view_config_md
from utils.load_preprocess import Preprocess
from config.help_messages import *


TITLE = "Preprocess data"


def main(config=None):
    st.markdown("""---""")
    if config:
        view_config_md(config)
    else:
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

