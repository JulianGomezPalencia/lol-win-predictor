"""Arranca Streamlit solo en la máquina local."""
from pathlib import Path
import sys
from streamlit.web import cli

if __name__ == "__main__":
    sys.argv = ["streamlit", "run", str(Path(__file__).with_name("screen.py")), "--server.address=127.0.0.1", "--browser.gatherUsageStats=false"]
    cli.main()
