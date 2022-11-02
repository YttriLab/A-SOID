import streamlit as st
from utils.project_utils import view_config_md
from utils.load_preprocess import Preprocess
from config.help_messages import IMPRESS_TEXT
from app import swap_app

import categories

CATEGORY = categories.PREPROCESS_DATA
TITLE = "Preprocess data"


def main(config=None):
    st.markdown("""---""")
    if config:
        view_config_md(config)
    else:
        processor = Preprocess()
        processor.main()

    bottom_cont = st.container()
    with bottom_cont:

        st.markdown("""---""")
        st.write('')
        button_col1, button_col2, button_col3, button_col4, button_col5 = st.columns([3, 3, 1, 1, 1])
        if button_col1.button('◀  PRIOR STEP'):
            swap_app('A-data-preprocess')
        if button_col5.button('NEXT STEP ▶'):
            swap_app('B-extract-features')
        st.write('')
        st.markdown('<span style="color:grey">{}</span>'.format(IMPRESS_TEXT), unsafe_allow_html=True)


if __name__ == "__main__":
    main()

