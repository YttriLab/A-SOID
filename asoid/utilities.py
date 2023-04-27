import streamlit as st
import base64
from streamlit.components.v1 import html
import time


import pathlib
from typing import BinaryIO, Union

File = Union[pathlib.Path, str, BinaryIO]


class SessionState:
    def __init__(self, state):
        for key, val in state.items():
            setattr(self, key, val)

    def update(self, state):
        for key, val in state.items():
            try:
                getattr(self, key)
            except AttributeError:
                setattr(self, key, val)


def get_session_id() -> str:
    ctx = st.scriptrunner.add_script_run_ctx()
    session_id: str = ctx.streamlit_script_run_ctx.session_id

    return session_id


def get_session(session_id: str = None):
    if session_id is None:
        session_id = get_session_id()

    session_info = st.server.server.Server.get_current()._get_session_info(session_id)

    if session_info is None:
        raise ValueError("No session info found")

    report_session = session_info.session

    return report_session


def session_state(**state):
    session = get_session()

    try:
        session._custom_session_state.update(state)
    except AttributeError:
        session._custom_session_state = SessionState(state)

    return session._custom_session_state
