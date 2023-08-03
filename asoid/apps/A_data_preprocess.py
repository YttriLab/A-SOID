import streamlit as st
from apps import A_data_preprocess, B_extract_features
from utils.project_utils import view_config_md
from utils.load_preprocess import Preprocess
from config.help_messages import IMPRESS_TEXT

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
    st.session_state['page'] = 'Step 2'
    bottom_cont = st.container()
    with bottom_cont:

        st.markdown("""---""")
        st.write('')
        # button_col1, button_col2, button_col3, button_col4, button_col5 = st.columns([3, 3, 1, 1, 1])
        # if button_col1.button('◀  PRIOR STEP'):
        #     # A_data_preprocess.main()
        #     pass
        # if button_col5.button('NEXT STEP ▶'):
        #     B_extract_features.main()
        st.write('')
        st.markdown('<span style="color:grey">{}</span>'.format(IMPRESS_TEXT), unsafe_allow_html=True)


if __name__ == "__main__":
    main()

