# spectrum-interpolation-for-python
The Python implementation of spectrum interpolation.

```python
raw = men.io.read_raw_fif(raw_path,preload=True)
Fl = [50]
dftbandwidth = [2]
dftneighbourwidth = [2]
raw_interpolation = spectrum_interpolation(raw,Fl, dftbandwidth, dftneighbourwidth)
```

reference to  S. Leske and S. S. Dalal, “Reducing power line noise in EEG and MEG data via spectrum interpolation,” Neuroimage, vol. 189, pp. 763–776, Apr. 2019, doi: 10.1016/j.neuroimage.2019.01.026.
