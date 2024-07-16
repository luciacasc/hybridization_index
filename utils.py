import numpy as np




def retrieve_atom_radius(atom_num)->float:
    """Input atomic number to return the covalent radius of atom.


    Args:
        atom_num (_type_): atomic number.

    Returns:
        int: covalent radius of atom (Bohr).
    """
    import periodictable as pt
    atom = pt.elements[atom_num]
    # factor 1.89 is the conversion factor angstrom to bohr
    return (atom.covalent_radius*1.89)





def create_exp(x, sigma)-> float:
    """
    Calculate the exponential function for a given value `x` and parameter `sigma`.

    The function computes the value of the exponential expression `exp(-0.5 * x / sigma^2)`.

    Args:
        x (float or np.ndarray): The input value(s) for which the exponential is calculated. 
                                 Can be a single float or a NumPy array of floats.
        sigma (float): The parameter `sigma` which scales the input `x`. Must be a positive float.

    Returns:
        float or np.ndarray: The calculated exponential value(s). If `x` is a single float, 
                             the return is a single float. If `x` is a NumPy array, the return 
                             is a NumPy array of floats with the same shape as `x`.
    """

    return np.exp(-0.5 * x / sigma**2, dtype=np.float64) 


