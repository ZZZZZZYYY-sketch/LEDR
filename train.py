"""Convenience entry point for the supplied LEDR/DBLK research code.

This wrapper executes mainCKD.py exactly as the original archive does. All command-line
arguments are defined in mainCKD.py.
"""
import runpy

if __name__ == "__main__":
    runpy.run_module("mainCKD", run_name="__main__")
