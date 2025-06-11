import streamlit as st
from asoid.config.help_messages import *
from asoid.utils.manual_refinement import Refinement



TITLE = "Refine behaviors"
REFINE_HELP = ("After using the :orange[Active Learning] step, you can further refine your model by adding new, previously unseen data (video + pose)."
               " **The added data does not require manual labels**, but will be classified by the current model.\n\n"
               " Similiar to the active learning regime, the model will suggest samples with low-confidence (or random) that can be refined manually in this step."
               " The refined data will then be used to retrain the model and improve its performance (in the next step). \n\n"
               " :blue[You can add multiple sessions (video + pose) before retraining the model.] \n\n"
               " ---\n\n"
                " **Step 1**: Select an iteration*."
               "\n\n **Step 2**: Select a refinement or add a new video.\n\n"
               " **Step 3**: Go through each behavior (radio buttons) and each example (tabs) and refine the predicted labels."
               "\n\n **Step 4**: Save the refined data before moving to the next step or a new example."
               "\n\n **Step 5**: Go to the next tab :orange[Create New Dataset] and retrain the model with the new data."
               "\n\n :blue[This process can be repeated for several iterations (requires new refinements).]"
                "\n\n:grey[* This refers to the iterations of refinement, followed by active learning.]"
               "\n\n ---\n\n"
               ":red[**Don't forget to save the refined data before moving to the next step or a new session.**]"
               )


def main(ri=None, config=None):
    st.markdown("""---""")
    st.title("Refine Behaviors (optional)")
    st.expander("What is this?", expanded=False).markdown(REFINE_HELP)

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
