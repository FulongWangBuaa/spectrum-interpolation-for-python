# Spectrum interpolation for python
The Python implementation of spectrum interpolation.

# Quick start
```python
raw = men.io.read_raw_fif(raw_path,preload=True)
Fl = [50]
dftbandwidth = [2]
dftneighbourwidth = [2]
raw_interpolation = spectrum_interpolation(raw,Fl, dftbandwidth, dftneighbourwidth)
```

# Acknowledgements
- [1] Leske S., Dalal S. S. Reducing Power Line Noise in EEG and MEG Data via Spectrum Interpolation[J]. NeuroImage, 2019,189: 763-776. [DOI: https://doi.org/10.1016/j.neuroimage.2019.01.026](https://doi.org/10.1016/j.neuroimage.2019.01.026)
