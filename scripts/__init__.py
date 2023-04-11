"""
OptiMS
Optimisation mass spectrometry is a program designed to automate optimisation and/or parameter alteration in
mass spectrometry (MS). It is primarily designed for use with Waters MassLynx
"""

from scripts.optims import run_optims
from scripts.chrom_grab import chrom_grab
from scripts.mz_grab import mz_grab
from scripts.wash import run_wash
