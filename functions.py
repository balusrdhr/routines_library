import numpy as np
import h5py as h5
from dragons import meraxes, munge
import astropy.units as u, astropy.constants as C

def subvolumes(fname, snapshot, grid_name, vol_factor, fluctuation = True, whole_grid = False, quiet=True):
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

    whole_grid : bool
        Return the whole grid as well.

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

    arr_shape = int(grid.shape[0]/factor)

    ### Black Magic!! Took me a solid couple of hours to test and figure out. SHOULD work! ###
    for ii in np.arange(vol_factor):

        if subvols[ii].shape[2] > arr_shape:
            subvols[ii] = np.delete(subvols[ii], 2, axis=2)
        
        if subvols[ii].shape[1] > arr_shape:
            subvols[ii] = np.delete(subvols[ii], 2, axis=1)
        
        if subvols[ii].shape[0] > arr_shape:
            subvols[ii] = np.delete(subvols[ii], 2, axis=0)

    if not quiet:
        print(f"Each subvolume has the shape {subvols[0].shape}")

    if fluctuation:
        for ii in np.arange(vol_factor):
            subvols[ii] = subvols[ii]/subvols[ii].mean() - 1
        
        if not quiet:
            print(f"All the grids are now zero-mean quantities, i.e. fluctations in the \"{grid_name}\" field")

    if fluctuation and whole_grid:
        grid = grid/grid.mean() - 1

    boxlength = meraxes.read_input_params(fname)["BoxSize"]
    sidelength = boxlength/factor

    if not quiet:

        print(f"BoxSize of the full simulation: {boxlength} Mpc")
        print(f"Size of the subvolume: {sidelength} Mpc, i.e. the sidelengths are 1/{factor} of the original simulation.")
    
    if whole_grid:
        return subvols, sidelength, grid
    else:
        return subvols, sidelength


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


def get_mhaloes(fname, snapshot, mass_type = "Mass_FOF"):
    """
    Return the halo masses. Mostly for making the Halo Mass Functions.
    Note: VELOCIraptor stores the masses with the little_h factored in.
    
    Parameters
    ----------
    fname : str
        Name of the hdf5 file.

    snapshot : int
        Snapshot value to read in the grid from.

    mass_type : str
        The mass grid to read in. Choose from {"Mass_tot", "Mass_FOF", "Mass_200mean", "Mass_200crit"}
        default = "Mass_FOF"

    Returns
    -------

    mhaloes : Numpy array
        An array of the requested halo masses. Units: h^-1 M_Sun

    boxsize : float
        The boxsize of the subvolumes. Units: h^-1 Mpc
    
    redshift : float
    	Redshift value corresponding to the input snapshot

    little_h : float
    	Value of the little_h where H0 = 100*little_h kim/s/Mpc. Units: Dimensionless
    """
    
    with h5.File(fname, "r") as hf:
        mass_unit = hf["Header/Units"].attrs["Mass_unit_to_solarmass"]
        boxsize = hf["Header/Simulation"].attrs["Period"]
        little_h = hf["Header/Simulation"].attrs["h_val"]

        mhaloes = np.asarray(hf["Snap_{:03d}/{}".format(snapshot, mass_type)])
        mhaloes = mhaloes * mass_unit * little_h #VELOCIraptor stores the masses with the little_h factored in.

        scale_factor = np.array(hf["Snap_{:03d}".format(snapshot)].attrs["scalefactor"])
        redshift = 1/scale_factor - 1

        return mhaloes, boxsize, redshift, little_h


def CosmicVarianceError(k, ps, k_edges, L_box):
   """
    Compute the cosmic variance errors for a given power spectrum.
    The formula: Var = Power /sqrt(N) where N is the number modes in the k-bin.
    A good derivation of the formula is given in Mondal, R., MNRAS 456, 1936–1947 (2016).

    Note: Assumes the volume is cubical. Also what I am computing and returning here is 
    strictly speaking the standard deviation and not variance. 

    Parameters
    ----------
    k : Numpy array
	Array of k-values. This is the average k-value in the bin. 

    ps : Numpy array
	Array of power values. This is the average power in the k-bin. 

    k_edges : Numpy array
	Array of k-values that are the bin-edges in k-space. len(k_edges) = len(k) + 1
    
    L_box : float
	Side_length of the cubical volume.

    Returns
    -------

    CosmicVariance : Numpy array
        An array of the cosmic variance errors 

    """

   Dk = np.diff(k_edges) #bin-width
   N_c = 2*np.pi*k*k*Dk*((L_box/(2*np.pi))**3) #Number of modes in the bin. Note the 2 instead of the 4 in the beginning. Because of FFT of real field.
   CosmicVariance = ps/np.sqrt(N_c)
    
   return CosmicVariance


def Mags_to_Q(Mags, lambda_1 = 912, lambda_2 = 91, ):
   """
    Convert the LyC magnitude values from Meraxes to the ionising photon flux Q

    Note: 

    Parameters
    ----------
    Mags : Numpy array
	Array of Mags-values. 

    lambda_1 : float
    Lyman limit wavelength. (Default: 912 nm)
	Units doesn't matter as long as it is the same as lambda_2

    lambda_2 : float
    SED smallest wavelength. (Default: 91 nm)
	Units doesn't matter as long as it is the same as lambda_1
    
   Returns
   -------

   Q : Numpy array
   Array of ionising photon flux

   """

   f_n = 10**( (22.5 - mags) / 2.5) * 3.632e3 * u.nJy 
   f_n = np.float64(f_n).to(u.erg/u.s/u.cm**2/u.Hz)
   f_n = f_n * np.log(lambda1/lambda2)/ C.h.to(u.erg/u.Hz)
   f_n = f_n * 4 * np.pi * ( (10 * u.pc).to(u.cm) )**2

   return f_n
