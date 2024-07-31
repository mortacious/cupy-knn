try:
    from importlib.metadata import version

    __version__ = version("cupy_knn")
except Exception:  # pragma: no cover # pylint: disable=broad-exception-caught
    try:
        from ._version import __version__, __version_tuple__
    except ImportError:
        __version__ = '0.0.0'
        __version_tuple__ = (0, 0, 0)
        
from .lbvh_index import LBVHIndex