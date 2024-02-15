import streamlit as st
from config.help_messages import *
from utils.manual_refinement import Refinement



TITLE = "Refine behaviors"

def main(ri=None, config=None):
    st.markdown("""---""")
    if config:
        refinement = Refinement(ri, config)
        refinement.main()
    else:
        st.error(NO_CONFIG_HELP)

    bottom_cont = st.container()
    with bottom_cont:

        st.markdown("""---""")
        st.write('')
        st.markdown('<span style="color:grey">{}</span>'.format(IMPRESS_TEXT), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
