import numpy as np
import h5py as h5
from dragons import meraxes, munge


def subvolumes(fname, snapshot, grid_name, vol_factor, fluctuation = True, quiet=True):
    """
    Split a 3D-grid of values into subvolumes.

    Parameters
    ----------
    fname : str
        Name of the hdf5 file.

    snapshot : int
        Snapshot value to read in the grid from.

    grid_name : str
        Name of the grid/field to read in.

    vol_factor : int
        The factor by which the volume is to be subdivided into.

    fluctuation : bool
        Whether the returned grid/field should be a fluctuation (zero-mean quantity) or not.
        Obtained by (grid/grid.mean - 1).
        (default = True)

    Returns
    -------

    Subvols : Numpy nd-array
        An array of the subvolume arrays. Shape: (vol_factor, grid.size/cbrt(vol_factor)*3).

    sidelength : float
        The boxsize of the subvolumes.
    """

    grid = meraxes.io.read_grid(fname, snapshot, grid_name)

    if not quiet:
        print(f"We are splitting the simulation volume into {vol_factor} equal parts.\n The shape of the input grid is {grid.shape}.")

    factor = int(np.cbrt(vol_factor))

    subvols = []
    first = np.array_split(grid, factor, axis = 0)

    for arr in first:
        sec = np.array_split(arr, factor, axis=1)

        for arr in sec:
            third = np.array_split(arr, factor, axis=2)
            subvols.extend(third)

    subvols = np.asarray(subvols, dtype=object)

    for ii in np.arange(vol_factor):
        if subvols[ii].shape[0] < subvols[ii].shape[1]:
            subvols[ii] = np.delete(subvols[ii], 0, axis=1)
        if subvols[ii].shape[0] > subvols[ii].shape[1]:
            subvols[ii] = np.delete(subvols[ii], 0, axis=0)

    #if not quiet:
     #   print(f"Each subvolume have the shape {subvols[0].shape}")

    if fluctuation:
        for ii in np.arange(vol_factor):
            subvols[ii] = subvols[ii]/subvols[ii].mean() - 1

        if not quiet:
            print(f"All the grids are now zero-mean quantities, i.e. fluctations in the \"{grid_name}\" field")

    boxlength = meraxes.read_input_params(fname)["BoxSize"]
    sidelength = boxlength/factor

    if not quiet:

        print(f"BoxSize of the full simulation: {boxlength} Mpc")
        print(f"Size of the subvolume: {sidelength} Mpc, i.e. the sidelengths are 1/{factor} of the original simulation.")

    return subvols, sidelength

#############################################################################################
#############################################################################################
#############################################################################################

def check_for_xH(fname, xH_val, tol=0.05):
    """
    Check a Meraxes output file for the presence of a particular global xH.

    *Args*:
        fname : str
            Full path to input hdf5 master file

        xH_val : float
            xH value

    *Kwargs*:
        tol : float
            +- tolerance on xH value present.  An error will be thrown if
            no xH within this tollerance is found.

    *Returns*:
        snapshot : int
            Closest snapshot

        redshift : float
            Closest corresponding redshift

        xH : float
            Neutral fraction of the snapshot
    """

    snaps, z, _ = meraxes.io.read_snaplist(fname)
    xH = meraxes.io.read_global_xH(fname, snaps, weight="volume")
    xHs = xH - xH_val

    w = np.argmin(np.abs(xHs))

    if np.abs(xHs[w]) > tol:
        raise KeyError("No xH value within tolerance found.")

    return int(snaps[w]), np.round(z[w], 2), np.round(xH[w], 2)

#############################################################################################
#############################################################################################
#############################################################################################

