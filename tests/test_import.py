import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import astrodynx as adx

def test_import():
    assert adx is not None

def test_version():
    assert hasattr(adx, "__version__")
    assert isinstance(adx.__version__, str)
    assert len(adx.__version__) > 0
