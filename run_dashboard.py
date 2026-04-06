#!/usr/bin/env python3
"""
Quick launcher script for the Sector War Graph dashboard.
Run: python3 run_dashboard.py
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from viz.dashboard import run_dashboard

if __name__ == '__main__':
    print('\n' + '=' * 60)
    print('  Sector War Graph — Interactive Dashboard')
    print('  http://127.0.0.1:8050')
    print('=' * 60 + '\n')
    run_dashboard()
