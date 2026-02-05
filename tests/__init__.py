#!/usr/bin/env python3
import sys
from pathlib import Path

# Add src folders to sys.path once for all tests
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
sys.path.append(str(Path(__file__).resolve().parent.parent / "production_scripts"))
# TODO Why is not working for all tests files