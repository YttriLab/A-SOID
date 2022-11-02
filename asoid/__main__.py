import click
import streamlit.cli
import os

@click.group()
def main():
    pass

@main.command("app")
def main_streamlit():
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, 'app.py')
    args = []
    streamlit.cli._main_run(filename, args)

if __name__ == "__main__":
    main()