import categories
import streamlit as st
from app import swap_app
from config.help_messages import IMPRESS_TEXT, NO_CONFIG_HELP
from utils.unsupervised_discovery import Explorer


CATEGORY = categories.DISCOVER
TITLE = "Unsupervised discovery"

def main(config=None):
    st.markdown("""---""")
    if config is not None:
        explorer = Explorer(config)
        explorer.main()
    else:
        st.error(NO_CONFIG_HELP)
    bottom_cont = st.container()
    with bottom_cont:

        st.markdown("""---""")
        st.write('')
        button_col1, button_col2, button_col3, button_col4, button_col5 = st.columns([3, 3, 1, 1, 1])
        if button_col1.button('◀  PRIOR STEP'):
            swap_app('F-view')
        if button_col5.button('NEXT STEP ▶'):
            swap_app('A-data-preprocess')
        st.write('')
        st.markdown('<span style="color:grey">{}</span>'.format(IMPRESS_TEXT), unsafe_allow_html=True)