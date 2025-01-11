import pathlib
from dataclasses import dataclass, field
from typing import List, Tuple, Union

import astropy.units as un
import h5py
import numpy as np
from astropy.coordinates import Galactic, SkyCoord
from astropy.io import fits
from astropy.units import Quantity
from astropy.wcs import WCS
from astropy_healpix import HEALPix

@dataclass(init=True, repr=True)
class FaradaySky:
    filename: str = None
    nside: int = None
    ordering: str = None
    extension: str = field(init=False, default=None)
    data: Tuple[np.ndarray, np.ndarray] = field(init=False, default=None)
    hp: HEALPix = field(init=False, default=None)

    def __post_init__(self):
        if self.nside is None:
            self.nside = 512

        if self.ordering is None:
            self.ordering = "ring"

        if self.filename is None:
            self.filename = (
                pathlib.Path(__file__).parent.resolve() / "./faraday_sky_files/faraday2020v2.hdf5"
            )

        self.extension = pathlib.Path(self.filename).suffix
        if self.extension == ".hdf5":
            with h5py.File(self.filename, "r") as hf:
                self.data = (
                    np.array(hf.get("faraday_sky_mean")),
                    np.array(hf.get("faraday_sky_std")),
                )
        elif self.extension == ".fits":
            with fits.open(self.filename) as hdul:
                self.data = (
                    hdul[1].data["faraday_sky_mean"],
                    hdul[1].data["faraday_sky_std"],
                )
                self.ordering = hdul[0].header.get("ORDERING", self.ordering)
                self.nside = hdul[0].header.get("NSIDE", self.nside)
        else:
            raise ValueError("The extension is not HDF5 or FITS")

        if self.nside and self.ordering:
            self.hp = HEALPix(nside=self.nside, order=self.ordering, frame=Galactic())

    def galactic_rm(
        self,
        ra: Union[List[Quantity], Quantity, str] = None,
        dec: Union[List[Quantity], Quantity, str] = None,
        frame="icrs",
        use_bilinear_interpolation: bool = False,
    ):
        ra, dec = self._validate_coordinates(ra, dec)

        coord = SkyCoord(ra=ra, dec=dec, frame=frame)

        if use_bilinear_interpolation:
            rm_value_mean = (
                self.hp.interpolate_bilinear_skycoord(coord, self.data[0]) * un.rad / un.m**2
            )
            rm_value_std = (
                self.hp.interpolate_bilinear_skycoord(coord, self.data[1]) * un.rad / un.m**2
            )
        else:
            healpix_idx = self.hp.skycoord_to_healpix(coord)
            rm_value_mean = self.data[0][healpix_idx] * un.rad / un.m**2
            rm_value_std = self.data[1][healpix_idx] * un.rad / un.m**2

        return rm_value_mean, rm_value_std

    def galactic_rm_image(
        self,
        fitsfile: Union[str, fits.HDUList, fits.PrimaryHDU, fits.Header] = None,
        use_bilinear_interpolation: bool = False,
    ):
        header = self._extract_header(fitsfile)
        w = WCS(header, naxis=2)

        m, n = header["NAXIS1"], header["NAXIS2"]
        frame = header["RADESYS"].lower()

        x, y = np.arange(m), np.arange(n)
        xx, yy = np.meshgrid(x, y)

        skycoord = w.array_index_to_world(xx, yy)

        rm_flattened = self.galactic_rm(
            ra=skycoord.ra.ravel(),
            dec=skycoord.dec.ravel(),
            frame=frame,
            use_bilinear_interpolation=use_bilinear_interpolation,
        )
        rm_mean = rm_flattened[0].reshape(n, m)
        rm_std = rm_flattened[1].reshape(n, m)

        print(
            f"The Galactic RM in the field is {rm_mean.mean():.2f} ± {rm_std.mean():.2f}"
        )
        return rm_mean, rm_std

    def _validate_coordinates(self, ra, dec):
        if isinstance(ra, Quantity):
            ra = ra.to(un.rad)
        elif isinstance(ra, str):
            ra = Quantity(ra)
        else:
            raise TypeError("Not valid type for RA")

        if isinstance(dec, Quantity):
            dec = dec.to(un.rad)
        elif isinstance(dec, str):
            dec = Quantity(dec)
        else:
            raise TypeError("Not valid type for DEC")

        return ra, dec

    def _extract_header(self, fitsfile):
        if isinstance(fitsfile, str):
            return fits.getheader(fitsfile)
        elif isinstance(fitsfile, (fits.HDUList, fits.PrimaryHDU)):
            return fitsfile[0].header
        elif isinstance(fitsfile, fits.Header):
            return fitsfile
        else:
            raise TypeError("Invalid FITS file type")
