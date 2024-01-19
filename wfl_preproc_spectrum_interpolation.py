import numpy as np
import mne

def wfl_preproc_spectrum_interpolation(raw,exclude, Fl, dftbandwidth, dftneighbourwidth):
    '''
    FT_PREPROC_DFTFILTER reduces power line noise (50 or 60Hz) using a 
    % discrete Fourier transform (DFT) filter, or spectrum interpolation.
    %
    % Use as
    %   [filt] = ft_preproc_dftfilter(dat, Fsample)
    % or 
    %   [filt] = ft_preproc_dftfilter(dat, Fsample, Fline)
    % or
    %   [filt] = ft_preproc_dftfilter(dat, Fsample, Fline, 'dftreplace', 'zero')
    % or
    %   [filt] = ft_preproc_dftfilter(dat, Fsample, Fline, 'dftreplace', 'neighbour')
    % where
    %   dat        data matrix (Nchans X Ntime)
    %   Fsample    sampling frequency in Hz
    %   Fline      frequency of the power line interference (if omitted from the input
    %              the default European value of 50 Hz is assumed)
    % 
    % Additional optional arguments are to be provided as key-value pairs:
    %   dftreplace = 'zero' (default), 'neighbour' or 'neighbour_fft'.
    % 
    %
    % If dftreplace = 'neighbour_fft' the powerline component is reduced via spectrum 
    % interpolation, in its original implementation, based on an algorithm that
    % uses an FFT and iFFT for the estimation of the spectral components. The signal is:
    % I)   transformed into the frequency domain via a fast Fourier
    %       transform (FFT),
    % II)  the line noise component (e.g. 50Hz, dftbandwidth = 1 (±1Hz): 49-51Hz) is
    %       interpolated in the amplitude spectrum by replacing the amplitude
    %       of this frequency bin by the mean of the adjacent frequency bins
    %       ('neighbours', e.g. 49Hz and 51Hz).
    %       dftneighbourwidth defines frequencies considered for the mean (e.g.
    %       dftneighbourwidth = 2 (±2Hz) implies 47-49 Hz and 51-53 Hz).
    %       The original phase information of the noise frequency bin is
    %       retained.
    % III) the signal is transformed back into the time domain via inverse FFT
    %       (iFFT).
    % If Fline is a vector (e.g. [50 100 150]), harmonics are also considered.
    % Preferably the data should be continuous or consist of long data segments
    % (several seconds) to avoid edge effects. If the sampling rate and the
    % data length are such, that a full cycle of the line noise and the harmonics
    % fit in the data and if the line noise is stationary (e.g. no variations
    % in amplitude or frequency), then spectrum interpolation can also be
    % applied to short trials. But it should be used with caution and checked
    % for edge effects.
    %
    % When using spectral interpolation, additional arguments are:
    %
    %   dftbandwidth      half bandwidth of line noise frequency bands, applies to spectrum interpolation, in Hz
    %   dftneighbourwidth width of frequencies neighbouring line noise frequencies, applies to spectrum interpolation (dftreplace = 'neighbour'), in Hz
    % 
    % If the data contains NaNs, the output of the affected channel(s) will be
    % all(NaN).
    %
    % See also PREPROC
    '''
    # defaults
    # 获取函数参数中的可选参数
    # dftbandwidth = kwargs.get('dftbandwidth', [2])
    # dftneighbourwidth = kwargs.get('dftneighbourwidth', [2])
    Fs = raw.info['sfreq']
    channels_name = raw.info['ch_names']
    bad_channel = raw.info['bads']
    raw_pick = raw.copy().drop_channels(bad_channel + exclude)
    dat = raw_pick.get_data()
    nchans, nsamples = dat.shape
    # 检查输入数据是否包含 NaN 值，如果存在则输出警告信息
    if np.any(np.isnan(dat)):
        print('data contains NaN values')
    # 将输入的频率Fl转换为一维数组
    Fl = np.atleast_1d(Fl)
    # 计算在给定采样率Fs下，能够容纳完整线噪声周期的整数个周期数
    n = np.round(np.floor(nsamples * (Fl / Fs + 100 * np.finfo(float).eps)) * Fs / Fl)
    # 如果数据不能在单个步骤中滤波，则递归地为每个频率执行滤波操作，并将结果级联处理
    if not np.all(n == n[0]):
        filt = dat.copy()
        for i in range(len(Fl)):
            filt = wfl_preproc_spectrum_interpolation(filt, Fs, Fl[i],dftbandwidth=dftbandwidth[i], dftneighbourwidth=dftneighbourwidth[i])
        return filt

    # Method: Spectrum Interpolation
    dftbandwidth = np.asarray(dftbandwidth)
    dftneighbourwidth = np.asarray(dftneighbourwidth)
    # 检查数据长度是否与线噪声频率的完整周期匹配
    if len(Fl) < len(dftbandwidth):
        dftbandwidth = dftbandwidth[:len(Fl)]
    if len(Fl) < len(dftneighbourwidth):
        dftneighbourwidth = dftneighbourwidth[:len(Fl)]

    if n != nsamples:
        raise ValueError('Spectrum interpolation requires that the data length fits complete cycles of the powerline frequency.')

    nfft = nsamples

    if len(Fl) != len(dftbandwidth) or len(Fl) != len(dftneighbourwidth):
        raise ValueError('The number of frequencies to interpolate should be the same as the number of bandwidths and neighbourwidths')
    # frequencies to interpolate
    f2int = np.array([Fl - dftbandwidth, Fl + dftbandwidth]).T

    # frequencies used for interpolation
    f4int = np.array([f2int[:, 0] - dftneighbourwidth, f2int[:, 0], f2int[:, 1], f2int[:, 1] + dftneighbourwidth]).T

    data_fft = np.fft.fft(dat, n=nfft, axis=1)

    frq = np.linspace(start = 0, stop = 1, num = nfft+1) * Fs

    # interpolate 50Hz (and harmonics) amplitude in spectrum
    for i in range(len(Fl)):
        # samples of frequencies that will be interpolated
        smpl2int = np.arange(np.argmin(np.abs(frq - f2int[i, 0])),
                                np.argmin(np.abs(frq - f2int[i, 1])) + 1)

        # samples of neighbouring frequencies used to calculate the mean
        low_neighbouring = np.arange(np.argmin(np.abs(frq - f4int[i, 0])),np.argmin(np.abs(frq - f4int[i, 1])))
        high_neighbouring = np.arange(np.argmin(np.abs(frq - f4int[i, 2])) + 1, np.argmin(np.abs(frq - f4int[i, 3])) + 1)

        smpl4int = np.concatenate((low_neighbouring,high_neighbouring))

        # new amplitude is calculated as the mean of the neighbouring frequencies
        mns4int = np.ones(data_fft[:,smpl2int].shape) * np.mean(np.abs(data_fft[:, smpl4int]), axis=1,keepdims=True)

        # Eulers formula: replace noise components with new mean amplitude combined with phase, that is retained from the original data
        data_fft[:, smpl2int] = np.exp(1j * np.angle(data_fft[:, smpl2int])) * mns4int
    # complex fourier coefficients are transformed back into time domin, fourier coefficients are treated as conjugate 'symmetric'
    # to ensure a real valued signal after iFFT
    half_len = data_fft.shape[1] // 2
    data_fft = np.column_stack((data_fft[:,0],data_fft[:,1:half_len],data_fft[:,half_len],np.conj(np.flip(data_fft[:,1:half_len],axis=1))))
    filt = np.fft.ifft(data_fft, axis=1)
    filt = filt.real
    
    raw_interpolation = mne.io.RawArray(filt, raw_pick.info)

    raw_interpolation.add_channels([raw.pick(bad_channel + exclude)])
    raw_interpolation.reorder_channels(channels_name)
    raw_interpolation.info['bads'] = bad_channel

    return raw_interpolation
