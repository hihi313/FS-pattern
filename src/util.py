from datetime import datetime
from pathlib import Path


def get_default_name(prefix:str=None, time_fmt: str = "%Y%m%d_%H%M%S") -> str:
    """
    Generate a default name based on the current datetime.

    Args:
        time_fmt (str): The format for the datetime string.

    Returns:
        str: A formatted string representing the current datetime.
    """
    return f"{prefix}{datetime.now().strftime(time_fmt)}"

def get_output_path(parent:str="./", filename:str=None, ext:str="jpg") -> str:
    parent = Path(parent)
    parent.mkdir(parents=True, exist_ok=True)
    if filename is None:
        filename = get_default_name(prefix="output_")
    return f"{parent / f'{filename}.{ext}'}"