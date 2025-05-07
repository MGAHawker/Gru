# Gru
This repository contains all code produced for the analysis of improvements made to the diamagnetic-levitator based gravimeter setup of the Ulbricht group within the University of Southampton. Mainly written by myself Michael Hawker for my BSc project, with major contributions from MPhys students Elizabeth Bondoc and Phoenix McArthur on the project. The camera based modal frequency identification code was produced by E. Bondoc MPhys. Ringdown analysis code was provided by PhD student Elliot Simcox of the Ulbricht group.

The trc file loading and base power spectral density production code has been taken from the Optoanalysis package created by Ashley Setter.

The code for specific stages is as follows:

PSDs- PSD code pm minor grid.py
A very slight adaptation of the PSD code found in optoanalysis, with minor axes added to aid in frequency value identification.

Stitching- stitching with window functions.py
Attempts ordering of a set of named trace files based on file creation times. Gaps between files are identified and filled with a zero signal which is faded into.

Butterworth bandpass filter step- working bandpass filter.py, Bandpass responsiveness.py
Simple SciPy based Butterworth bandpass filter, bandwidth and the frequency focused on can be specified. Produces short plots from the files loaded to visualise filtering.
Bandpass responsiveness tests effects of different orders on the filter shape.

Quadrature work- Quadrature initial.py
Sets up plotting the two detector arms against each other. A fitted ellipse and get closest point function could be added to make this viable for future Quadrature work on the setup.

Peak stability analysis- Peak stability full weekend working (error bars),py
Two versions made depending on whether error bars are requested, individual plots can be made false to prevent showing. Takes data sets and arranges them in absolute time, they're then chunked into a specified length (currently 250 seconds) and PSDs produced. A Lorentzian fit then extracts PSD peak data (frequency and amplitude) used for the stability plot.

Signal to noise ratio analysis- SNR analysis.py
Takes two files, such as from data sets before and after improvements to the gravimeter. The signal is specified as the power within a modal identification region manually entered. The noise is the average power of everything outside of this. An SNR is then calculated from the raw signal and noise powers and then converted into decibels. SNR improvments are then calculated by comparison between the two files SNRs in a dB and linear form. Noise reduction is calcualted by direct comparison of raw noise powers.

DC offset averaging plot with lunar model fitting implemented- DC averaging plot moon signal fitted with datetime axes implemented Alternative error bars.py
Not the most simple naming but I had a large amount of iterations producing this
A data set of files can be specified using their naming convention to be loaded and ordered in absolute time as done elsewhere in my code. Half hour bins are specified for the full loaded data set, populated and averaged to give a DC offset trace. A Savitsky-Golay filter is used to remove pressurisation trends that were present in our data. The "lunar component" is the DC offset trace with the baseline identified subtracted.
A simple lunar model based on the expected period of 12 hours 25 minutes is then fitted to the "lunar component".
FFT plots of the baseline and "lunar component" are produced.
A more comprehensive model of the lunar and solar signals is constructed using ephem and fitting again carried out. In the event a combined effect is observed a plot of the best combined effect should be produced.

Ringdown- Elliot's ringdown.py
Code provided by PhD student Elliot Simcox of the Ulbricht group to obtain Q factors from excited state ringdowns of the system.

