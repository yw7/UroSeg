from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("uroseg")
except PackageNotFoundError:
    __version__ = "0.0.0"
