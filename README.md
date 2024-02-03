# OptiMS

Optimization Mass Spectrometry (OptiMS) is a Pythonic application programming interface (API) which automates optimization and/or parameter alteration in mass spectrometry (MS). It was originally designed for use with Waters MassLynx but the code can be adapted to be used with a range of mass spectrometers and their software.

A pre-print describing the code and its applications is available at DOI:[10.26434/chemrxiv-2023-77wdm](https://chemrxiv.org/engage/chemrxiv/article-details/650b36a460c37f4f76244741). 

## Installation

The code can be downloaded as a package for offline use using the conventional github download tools.

## Usage

Tutorials videos on usage are a working progress.

### Main optimization and/or parameter alteration function

```python
# Import package
import optims

# Automate optimization and/or parameter alteration
optims.run_optims(**kwargs)
```

Optimization methods include one factor at a time 'OFAT', exhaustive 'exhaustive', and Bayesian optimization 'BO'. Further methods and other parameters are explained in the optims.py file.

### Adapting the code to different instruments and software

The code can be adapted in the instrument_map.py file. Here, the functions chrom_refresh_custom, get_chrom_custom, and stop_aq_custom can be edited to add the mouse and keyboard movements necessary to refresh chromgratograms, copy chromatograms, and stop the acquisition respectively. The software used in run_optims must then be specified as 'custom'.

### Further functionality specific to Waters MassLynx

```python
# Collect further OptiMS data after experiment completion
optims.optims_grab(**kwargs)

# Collect multiple chromatogramic data after experiment completion
optims.chrom_grab(**kwargs)

# Collect mz data from each region of experimental conditions after experiment completion
optims.mz_grab(**kwargs)
```

## Contributing

Ideally, OptiMS would include presets for various instruments and software. Therefore, feedback regarding the adaptations made by users to facilitate the interfacing of OptiMS with their instruments and software would be gratefully received.
