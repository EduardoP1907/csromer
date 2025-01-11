#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import dask.array as da
import numpy as np
from astropy.io import fits


def filter_cubes(data_I, data_Q, data_U, header, additional_outlier_idxs=None):
    init_freq = header["CRVAL3"]
    nfreqs = header["NAXIS3"]
    step_freq = header["CDELT3"]
    nu = init_freq + np.arange(0, nfreqs) * step_freq
    sum_I = np.nansum(data_I, axis=(1, 2))
    sum_Q = np.nansum(data_Q, axis=(1, 2))
    sum_U = np.nansum(data_U, axis=(1, 2))
    correct_freqs = np.where((sum_I != 0.0) | (sum_Q != 0.0) | (sum_U != 0.0))[0]
    if additional_outlier_idxs:
        correct_freqs = np.setxor1d(correct_freqs, additional_outlier_idxs)
    filtered_data = 100.0 * (nfreqs - len(correct_freqs)) / nfreqs
    print(f"Filtering {filtered_data:.2f}% of the total data")
    return (
        data_I[correct_freqs],
        data_Q[correct_freqs],
        data_U[correct_freqs],
        nu[correct_freqs],
    )


class Reader:

    def __init__(
        self,
        stokes_I_name=None,
        stokes_Q_name=None,
        stokes_U_name=None,
        Q_cube_name=None,
        U_cube_name=None,
        freq_file_name=None,
        numpy_file=None,
    ):
        self.stokes_I_name = stokes_I_name
        self.stokes_Q_name = stokes_Q_name
        self.stokes_U_name = stokes_U_name
        self.Q_cube_name = Q_cube_name
        self.U_cube_name = U_cube_name
        self.freq_file_name = freq_file_name
        self.numpy_file = numpy_file

    def readCube(self, file=None, stokes=None, memmap=True):
        file = file or (self.Q_cube_name if stokes == "Q" else self.U_cube_name)
        try:
            with fits.open(file, memmap=memmap) as hdu:
                print("FITS shape: ", hdu[0].data.squeeze().shape)
                return hdu[0].header, hdu[0].data.squeeze()
        except FileNotFoundError:
            sys.exit("FileNotFoundError: The FITS file cannot be found")

    def readQU(self, memmap=True):
        header, Q = self.readCube(self.Q_cube_name, memmap=memmap)
        _, U = self.readCube(self.U_cube_name, memmap=memmap)
        return Q, U, header

    def readImage(self, name=None, stokes=None):
        filename = name or (self.stokes_I_name if stokes == "I" else self.stokes_Q_name if stokes == "Q" else self.stokes_U_name)
        with fits.open(filename) as hdul:
            return hdul[0].header, np.squeeze(hdul[0].data)

    def readNumpyFile(self):
        try:
            np_array = np.load(self.numpy_file)
            return np_array[:, :, 0], np_array[:, :, 1]
        except FileNotFoundError:
            sys.exit("FileNotFoundError: The numpy file cannot be found")

    def readHeader(self, name=None):
        filename = name or self.Q_cube_name
        with fits.open(filename) as hdul_image:
            return hdul_image[0].header


class Writer:

    def __init__(self, output=""):
        self.output = output

    def writeFITSCube(self, cube, header, nphi, phi, dphi, output=None, overwrite=True):
        header.update({
            "NAXIS": 4,
            "NAXIS3": (nphi, "Length of Faraday depth axis"),
            "NAXIS4": (2, "Real and imaginary"),
            "CTYPE3": "Phi",
            "CDELT3": dphi,
            "CUNIT3": "rad/m/m",
            "CRVAL3": phi[0]
        })

        output_file = output or self.output

        if np.iscomplexobj(cube):
            real_part, imag_part = da.from_array(cube.real), da.from_array(cube.imag)
            concatenated_cube = da.stack([real_part, imag_part], axis=0)
        else:
            concatenated_cube = cube

        fits.writeto(output_file, data=concatenated_cube, header=header, overwrite=overwrite)

    def writeNPCube(self, cube, output=None):
        np.save(output or self.output, cube)

    def writeFITS(self, data=None, header=None, output=None, overwrite=True):
        hdu = fits.PrimaryHDU(data, header)
        hdu.writeto(output or self.output, overwrite=overwrite)
