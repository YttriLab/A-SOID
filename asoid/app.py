import configparser as cfg
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
from streamlit_option_menu import option_menu
import base64

from apps import *
from config.help_messages import UPLOAD_CONFIG_HELP, IMPRESS_TEXT
from utils.load_workspace import load_features, load_iterX


def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded


def img_to_html(img_path, width=500):
    img_html = f"<img src='data:image/png;base64,{img_to_bytes(img_path)}'  width='{width}px', class='img-fluid'>"
    return img_html


def index():
    """
    :param application_options: dictionary (app_key: apps)
    :return:
    """
    HERE = Path(__file__).parent.resolve()

    step1_fname = HERE.joinpath("images/data_process_wht.png")
    step2_fname = HERE.joinpath("images/feature_extraction_wht.png")
    step3_fname = HERE.joinpath("images/baseline_classifier_wht.png")
    step4_fname = HERE.joinpath("images/active_learning_schematic.png")
    step5_fname = HERE.joinpath("images/app_discovery.png")

    st.markdown(f" <h1 style='text-align: left; color: #f6386d; font-size:30px; "
                f"font-family:Avenir; font-weight:normal'>Welcome to A-SOiD</h1> "
                , unsafe_allow_html=True)
    st.write("---")
    st.markdown(f" <h1 style='text-align: left; color: #FFFFFF; font-size:18px; "
                f"font-family:Avenir; font-weight:normal'>"
                f"Introducing A-SOiD, the innovative no code website for building supervised classifiers. "
                f"With our platform, you can input pose and behavioral labels to create a customized classifier "
                f"that accurately identifies and classifies animal behavior. "
                f"Our active learning paradigm ensures balanced training data, "
                f"while our manual refinement process allows you to expand and refine behavioral classes "
                f"in an iterative fashion, 'learning the animal behavior as you go.'"
                f"And with our unsupervised segmentation technology, "
                f"A-SOiD can further dissect subtle differences within the same human-defined behavior, "
                f"providing even deeper insights into animal behavior."
                f"Best of all, A-SOiD's performance outperforms state-of-the-art solutions, "
                f"all without the need for a GPU. With in-depth analysis and interactive visuals, "
                f"as well as downloadable CSVs for easy integration into your existing workflow, "
                f"A-SOiD is the ultimate tool for building your own animal behavior classifiers. "
                f"Try A-SOiD today and unlock a new level of insights into animal behavior."
                , unsafe_allow_html=True)
    # st.write("---")
    st.markdown(f" <h1 style='text-align: left; color: #f6386d; font-size:18px; "
                f"font-family:Avenir; font-weight:normal'>Get Started by Selecting a Step</h1> "
                , unsafe_allow_html=True)



    selected_step = st.select_slider('',
                                     options=['Step 1',
                                              'Step 2',
                                              'Step 3',
                                              'Step 4',
                                              'Step 5'],
                                     value=st.session_state['page'],

                                     label_visibility='collapsed')


    colL, colR = st.columns(2)
    if selected_step == 'Step 1':
        colL.markdown(f" <h1 style='text-align: left; color: #FFFFFF; font-size:18px; "
                      f"font-family:Avenir; font-weight:normal'>Step 1: Upload Your Pose Estimation and Annotation Data"
                      , unsafe_allow_html=True)

        colL.markdown(f" <h1 style='text-align: left; color: #FFFFFF; font-size:18px; "
                      f"font-family:Avenir; font-weight:normal'>"
                      f""
                      f"To begin, ensure that your pose estimation files are in either DLC or SLEAP format. "
                      f"These files contain the spatial coordinates of the individual body parts in your video,"
                      f" enabling us to accurately track movement and analyze posture."
                      f"", unsafe_allow_html=True)
        colR.markdown("<p style='text-align: right; color: grey; '>" + img_to_html(step1_fname, width=350) + "</p>",
                      unsafe_allow_html=True)

    elif selected_step == 'Step 2':
        colL.markdown(f" <h1 style='text-align: left; color: #FFFFFF; font-size:18px; "
                       f"font-family:Avenir; font-weight:normal'> Step 2: Extract spatio-temporal features from pose"
                       f"", unsafe_allow_html=True)
        colL.markdown(f" <h1 style='text-align: left; color: #FFFFFF; font-size:18px; "
                       f"font-family:Avenir; font-weight:normal'> In this step,"
                       f" you will examine the distribution of your annotated behaviors. "
                       f"Once you define your minimum duration, features are then computed "
                       f"across time.", unsafe_allow_html=True)
        colR.markdown("<p style='text-align: right; color: grey; '>" + img_to_html(step2_fname, width=350) + "</p>",
                       unsafe_allow_html=True)
    elif selected_step == 'Step 3':
        colL.markdown(f" <h1 style='text-align: left; color: #FFFFFF; font-size:18px; "
                       f"font-family:Avenir; font-weight:normal'> Step 3: Training a classifier"
                       f"", unsafe_allow_html=True)
        colL.markdown(f" <h1 style='text-align: left; color: #FFFFFF; font-size:18px; "
                       f"font-family:Avenir; font-weight:normal'> In this step,"
                       f" you will build a machine learning classifier. "
                       f"A-SOiD automatically balances your training data to prevent  "
                       f"emphasis on large classes.", unsafe_allow_html=True)
        colR.markdown("<p style='text-align: right; color: grey; '>" + img_to_html(step3_fname, width=350) + "</p>",
                       unsafe_allow_html=True)
    elif selected_step == 'Step 4':
        colL.markdown(f" <h1 style='text-align: left; color: #FFFFFF; font-size:18px; "
                       f"font-family:Avenir; font-weight:normal'> Step 4: Manual refinement on new data"
                       f"", unsafe_allow_html=True)
        colL.markdown(f" <h1 style='text-align: left; color: #FFFFFF; font-size:18px; "
                       f"font-family:Avenir; font-weight:normal'> In this step,"
                       f" you will refine the low confidence behaviors. "
                       f"The refinements will be added to the training dataset."
                       f"", unsafe_allow_html=True)
        colR.markdown("<p style='text-align: right; color: grey; '>" + img_to_html(step4_fname, width=350) + "</p>",
                       unsafe_allow_html=True)

    elif selected_step == 'Step 5':
        colL.markdown(f" <h1 style='text-align: left; color: #FFFFFF; font-size:18px; "
                       f"font-family:Avenir; font-weight:normal'> Step 5: Discover subtle differences within behavior"
                       f"", unsafe_allow_html=True)
        colL.markdown(f" <h1 style='text-align: left; color: #FFFFFF; font-size:18px; "
                       f"font-family:Avenir; font-weight:normal'> In this step,"
                       f" you can run unsupervised learning on a particular behavior to get "
                       f"segmented behaviors."
                       f"", unsafe_allow_html=True)
        colR.markdown("<p style='text-align: right; color: grey; '>" + img_to_html(step5_fname, width=350) + "</p>",
                       unsafe_allow_html=True)
    # st.write("---")
    # st.markdown(f" <h1 style='text-align: center; color: #FFFFFF; font-size:18px; "
    #             f"font-family:Avenir; font-weight:normal'> Step 3: training a classifier"
    #             f"", unsafe_allow_html=True)
    # st.markdown("<p style='text-align: center; color: grey; '>" + img_to_html(step3_fname, width=300) + "</p>",
    #             unsafe_allow_html=True)
    # st.write("---")
    # st.markdown(f" <h1 style='text-align: center; color: #FFFFFF; font-size:18px; "
    #             f"font-family:Avenir; font-weight:normal'> Step 4: manual refinement on new data"
    #             f"", unsafe_allow_html=True)
    # st.markdown("<p style='text-align: center; color: grey; '>" + img_to_html(step4_fname, width=250) + "</p>",
    #             unsafe_allow_html=True)
    # st.write("---")
    # st.markdown(f" <h1 style='text-align: center; color: #FFFFFF; font-size:18px; "
    #             f"font-family:Avenir; font-weight:normal'> Step 5: discover subtle differences within behavior"
    #             f"", unsafe_allow_html=True)
    # st.markdown("<p style='text-align: center; color: grey; '>" + img_to_html(step5_fname, width=250) + "</p>",
    #             unsafe_allow_html=True)

    bottom_cont = st.container()

    with bottom_cont:
        st.markdown("""---""")
        st.write('')
        st.markdown('<span style="color:grey">{}</span>'.format(IMPRESS_TEXT), unsafe_allow_html=True)


def main():
    HERE = Path(__file__).parent.resolve()
    logo_fname_ = HERE.joinpath("images/asoid_logo.png")
    logo_im_ = Image.open(logo_fname_)
    # TODO: Decide on one, delete over
    logo_fname = HERE.joinpath("images/asoid_logo_wtext.png")
    logo_im = Image.open(logo_fname)
    # set webpage icon and layout
    st.set_page_config(
        page_title="A-SOiD",
        page_icon=logo_im_,
        layout="wide",
        menu_items={
        }
    )
    hide_streamlit_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                </style>
                """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"]{
            min-width: 250px;
            max-width: 250px;   
        }
        [data-testid="stSidebar"][aria-expanded="false"] {
            margin-left: -250px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    header_container = st.container()
    if 'page' not in st.session_state:
        st.session_state['page'] = 'Step 1'

    # if in main menu, display applications, see above index for item layout
    with st.sidebar:
        le, ri = header_container.columns([1, 1])

        uploaded_config = le.file_uploader('upload config file'.upper()
                                           , type='ini'
                                           , help=UPLOAD_CONFIG_HELP)

        if uploaded_config is not None:
            # Make temp file path from uploaded file
            with tempfile.NamedTemporaryFile(mode="wb", delete=False) as temp:
                bytes_data = uploaded_config.getvalue()
                temp.write(bytes_data)
            project_config = cfg.ConfigParser()
            project_config.optionxform = str
            with open(temp.name) as file:
                project_config.read_file(file)

        _, mid_im, _ = st.columns([0.35, 1, 0.35])
        _, mid_im2, _ = st.columns([0.415, 1, 0.415])
        mid_im.image(logo_im_)
        if mid_im2.button("Start A-SOiD"):
            A_data_preprocess.main()
        st.write('---')
        try:
            sections = [x for x in project_config.keys() if x != "DEFAULT"]
            for parameter, value in project_config[sections[0]].items():
                if parameter == 'PROJECT_PATH':
                    working_dir = value
                elif parameter == 'PROJECT_NAME':
                    prefix = value
                elif parameter == 'CLASSES':
                    annotations = value
            menu_options = ['Menu', 'Upload Data ✔', 'Extract Features', 'Active Learning',
                            'Refine Behaviors', 'Predict', "View", "Discover"]
            try:
                [features_train, _, _, _] = load_features(working_dir, prefix)
                menu_options = ['Menu', 'Upload Data ✔', 'Extract Features ✔', 'Active Learning',
                                'Refine Behaviors', 'Predict', "View", "Discover"]
                try:
                    [_, _, _, scores, _, _] = load_iterX(working_dir, prefix)
                    menu_options = ['Menu', 'Upload Data ✔', 'Extract Features ✔', 'Active Learning ✔',
                                    'Refine Behaviors', 'Predict', "View", "Discover"]
                except:
                    pass

            except:
                pass

        except:
            menu_options = ['Menu', 'Upload Data', 'Extract Features', 'Active Learning',
                            'Refine Behaviors', 'Predict', "View", "Discover"]

        app_names = np.array(['index',
                              'A-data-preprocess',
                              'B-extract-features',
                              'C-auto-active-learning',
                              'D-manual-active-learning',
                              'E-predict',
                              'F-view',
                              'G-unsupervised-discovery',
                              ])
        icon_options = ['window-desktop',
                        'upload',
                        'bar-chart-line',
                        'diagram-2',
                        'images',
                        'file-earmark-plus',
                        'binoculars',
                        "signpost-split",
                        ]
        # print(np.where(app_names == get_url_app())[0])
        nav_options = option_menu(None, menu_options,
                                  icons=icon_options,
                                  menu_icon="",
                                  default_index=0,
                                  # orientation="horizontal",
                                  styles={
                                      "container": {"padding": "0!important",
                                                    "background-color": "#131320",
                                                    },
                                      "icon": {"color": "white",
                                               "font-size": "20px"},
                                      "nav-link": {
                                          "font-size": "15px",
                                          "text-align": "left",
                                          "margin": "2px",
                                          "--hover-color": "#000000"},
                                      "nav-link-selected": {
                                          "font-weight": "normal",
                                          # "text-decoration": "underline",
                                          "color": "#FFFFFF",
                                          "background-color": '#f6386d'},
                                  }
                                  )
        # for i in range(len(menu_options)):
        #     if nav_options == menu_options[i]:
        #         swap_app(app_names[i])
    if nav_options == 'Menu':
        index()
    elif 'Upload Data' in nav_options:
        try:
            A_data_preprocess.main(config=project_config)
        except:
            A_data_preprocess.main(config=None)
    elif 'Extract Features' in nav_options:
        try:
            B_extract_features.main(config=project_config)
        except:
            B_extract_features.main(config=None)
    elif 'Active Learning' in nav_options:
        try:
            C_auto_active_learning.main(config=project_config)
        except:
            C_auto_active_learning.main(config=None)
    elif 'Refine Behaviors' in nav_options:
        try:
            D_manual_active_learning.main(config=project_config)
        except:
            D_manual_active_learning.main(config=None)
    elif 'Predict' in nav_options:
        try:
            E_predict.main(config=project_config)
        except:
            E_predict.main(config=None)
    elif 'View' in nav_options:
        try:
            F_view.main(config=project_config)
        except:
            F_view.main(config=None)
    elif 'Discover' in nav_options:
        try:
            G_unsupervised_discovery.main(config=project_config)
        except:
            G_unsupervised_discovery.main(config=None)

    # if session_state.app == "index":
    #     application_function = functools.partial(
    #         index, application_options=application_options,
    #     )
    #
    # else:
    #     try:
    #         application_function = functools.partial(
    #             application_options[session_state.app].main, config=project_config,
    #         )
    #     except:
    #         application_function = functools.partial(
    #             application_options[session_state.app].main, config=None,
    #         )

    # application_function()


if __name__ == "__main__":
    main()
