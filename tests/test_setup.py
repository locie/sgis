#!/usr/bin/env python3
import sys
from pathlib import Path

TEST_TARGET_FOLDER="/tmp/"

# Add src folders to sys.path once for all tests
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
sys.path.append(str(Path(__file__).resolve().parent.parent / "production_scripts"))