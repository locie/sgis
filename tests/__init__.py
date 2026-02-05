#!/usr/bin/env python3
import sys
from pathlib import Path

# Add src folder to sys.path once for all tests
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
