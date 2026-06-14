import sys
import os

# Add project root to path so 'neuroscan' package is importable during pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
