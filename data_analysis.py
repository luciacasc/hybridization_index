import numpy as np
import collections
import utils 



def dict_theo_charge(atom_type:np.array, charge:np.array)->dict: 
    """
    Calculate the theoretical charge for each type of atom.

    Parameters:
    atom_type (np.ndarray): Array containing atom types.
    charge (np.array): Array containing atomic charge.
    Returns:
    dict: Dictionary with atom types as keys and theoretical charges as values.
    """
    charge_unique, counts = np.unique(charge, return_counts= True)
    atom_type_unique = np.unique(atom_type)
    total_charge4atom = [charge_unique[i]* counts[i] for i in range(len(atom_type_unique)) ]
    
    return {key: total_charge4atom[i] for i,key in enumerate(atom_type_unique)}




def dict_computed_charge(atom_type: np.array, charge: np.array, charge4atom: np.array) -> dict:
    """
    Compute the total charge for each unique atom type.

    Parameters:
    atom_type (np.array): Array of atom types.
    charge (np.array): Array of charges for each atom type (this value is taken from the file).
    charge4atom (np.array): A 1D NumPy array containing the partial charge contributions for each atom.

    Returns:
    dict: Dictionary with atom types as keys and total charge as values.
    """
    # Combine atom type, charge, and density into a single matrix
    matrix1 = np.column_stack((atom_type, charge, charge4atom))
    
    # Count the number of occurrences of each atom type
    atom_counts = collections.Counter(matrix1[:, 0])
    
    # Calculate the total charge for each unique atom type
    comp_charge = {atom: sum(matrix1[matrix1[:, 0] == atom, 2]) for atom in atom_counts.keys()}
    
    return comp_charge








def charge_difference(dict_theoretical_charge:dict, dict_calculated_charge:dict) -> float:
    """
    Calculate the difference between theoretical and found charges for each atom type.


    Parameters:
    dict_theoretical_charge (dict): Dictionary of theoretical charges.
    dict_calculated_charge (dict): Dictionary of found charges.

    Returns:
    float: Errors ( = Sum of abs differences) between theoretical and calculated charges.
    """
    return sum(abs(dict_theoretical_charge[key] - dict_calculated_charge[key]) for key in list(dict_theoretical_charge.keys()))






def optimization_radius(atom_type: np.array, atom_coordinates: np.array, charge : np.array, density_grid : np.array, density_value : np.array, volume: float, radii : np.array):
    """
    Optimize the radius by finding the weight that minimizes the difference between theoretical and computed charges.

    Parameters:
    atom_type (np.array): Array of atom types.
    atom_coordinates (np.array): A 2D NumPy array of shape (n_atoms, 3) representing the x, y, z coordinates of the atoms.
    charge (np.array): Array of charges for each atom type (this value is taken from the file).
    density_grid (np.array): A 2D NumPy array of shape (n_points, 3) representing the coordinates of the electron density grid.
    density_value (np.array): A 1D NumPy array of shape (n_points,) containing the electron density values at the grid points.
    volume (float): The volume element used to scale the calculated charge.
    radii (np.array): A 1D NumPy array of shape (n_atoms,) representing the radii of the atoms.
        
    Returns:
    float: The weight that minimizes the charge difference.
    """
    
    weights = np.arange(0.5, 0.95, 0.05)

    dict_theoretical_charge = dict_theo_charge(atom_type, charge)
    list_dict_calc = []
    dict_calc_charge ={}
    errors_list = []

    for weight in weights:
        charge_atom_type, _ = charge4atom(atom_coordinates, density_grid, density_value, volume, radii, weight, False)
        
        dict_calc_charge = dict_computed_charge(atom_type, charge, charge_atom_type)
        list_dict_calc.append(dict_calc_charge)
        error = charge_difference(dict_theoretical_charge, dict_calc_charge)
        errors_list.append(error)


    # minimum_error = min(errors_list)
    idx_min = errors_list.index(min(errors_list))
    best_charge = list_dict_calc[idx_min]
    print(best_charge)
    return weights[idx_min]








def charge4atom(atom_coordinates:np.array, density_grid:np.array, density_value:np.array, volume:float, radii: np.array, weight:float, verbosity:bool):   
    """
    Calculate the partial charge contribution for a set of atoms based on their coordinates and electron density grid.

    This function computes the partial charge for each atom by evaluating the distance between the atom's coordinates
    and each point in the electron density grid, applying an exponential function to these distances, and then 
    calculating the charge contribution.

    Args:
        atom_coordinates (np.array): A 2D NumPy array of shape (n_atoms, 3) representing the x, y, z coordinates of the atoms.
        density_grid (np.array): A 2D NumPy array of shape (n_points, 3) representing the coordinates of the electron density grid.
        density_value (np.array): A 1D NumPy array of shape (n_points,) containing the electron density values at the grid points.
        volume (float): The volume element used to scale the calculated charge.
        radii (np.array): A 1D NumPy array of shape (n_atoms,) representing the radii of the atoms.
        weight (float): A scaling factor applied to the radii.
        verbosity (bool): If True, the function will print the partial charge contribution.

    Returns:
        np.array: A 1D NumPy array containing the partial charge contributions for each atom.
    """
    x,y,z = atom_coordinates[:, 0], atom_coordinates[:, 1], atom_coordinates[:, 2]
    xg,yg,zg = density_grid[:,0], density_grid[:,1], density_grid[:,2] 

    distance = np.array([(xg - x[i])**2 + (yg - y[i])**2 + (zg - z[i])**2 for i in range(len(atom_coordinates))], dtype=np.float64)
    
    sigma = np.array(radii*weight)

    # Initialize an array to store the results
    results = np.zeros_like(distance)
    for i in range(distance.shape[0]):
        results[i] = utils.create_exp(distance[i], sigma[i])

    charge4atom = np.array( (np.dot(results , abs(density_value)))*volume , dtype=np.float64)

    total_charge = sum(charge4atom)

    if verbosity:
        print('Partial charge contribution (for atom):\n')
        print(charge4atom)
    
    return(charge4atom, total_charge)
    
    





def hybridization_index(dens_metal_array:float,dens_molecule_array:float)->float:
    """This function calculates the hybridization index.

    Args:
        dens_metal_array (float):  Electronic charge of metal cluster.
        dens_molecule_array (float): Electronic charge of each molecule.

    Returns:
        float: Hybridization index 
    """
    #print('only for stilbene')
    # ground 01 {47.0: 378.33113097799196, 7.0: 9.081800286427898, 6.0: 51.112323868764534, 1.0: 1.4708248597871925}
    # ground 05 {47.0: 378.0526821936735, 7.0: 11.744120830444782, 6.0: 49.08506258244555, 1.0: 1.4042087037030169}
    # ground 09 {47.0: 377.53958049486994, 7.0: 7.732564554900119, 6.0: 55.728703906957605, 1.0: 1.3463409538880533}
    # ground 19 {47.0: 376.4138356782385, 7.0: 8.838138161279844, 6.0: 48.39523862858246, 1.0: 1.2841508603770095}
    # ground 25 {47.0: 380.1505546479942, 7.0: 8.620915991334623, 6.0: 49.61653270385586, 1.0: 1.4993273278221195}
    # ground 32 {47.0: 380.7140905738815, 7.0: 8.693287856461648, 6.0: 50.96028957772728, 1.0: 1.5433509254408448}
    # hybr stilb 2 mol {47.0: 404.4654405166957, 6.0: 156.70409398319757, 1.0: 4.147731482181722}
    # hybr stilb 1 mol hi = (dens_molecule_array/(58.999329444223484+ 1.7532056047643365))/(dens_metal_array/367.894492545562)

    #1 stil
    #hi = (dens_molecule_array/(9.081800286427898+51.112323868764534+1.4708248597871925))/(dens_metal_array/378.33113097799196)
    # 2 stil
    #hi = (dens_molecule_array/( 156.70409398319757*0.5+4.147731482181722*0.5))/(dens_metal_array/378.33113097799196)
    # 19
    #hi = (dens_molecule_array/(8.838138161279844+ 48.39523862858246+ 1.2841508603770095))/(dens_metal_array/376.4138356782385)
    #01
    hi = (dens_molecule_array/(9.081800286427898+ 51.112323868764534+1.4708248597871925))/(dens_metal_array/376.4138356782385)
    return(round(hi,2))





def ct_finder(density_grid:np.array, density_value:np.array)->float:
    """
    Calculate the D index, which is a measure of spatial separation of positive and negative densities.

    Args:
        density_grid (np.array): A N x 3 array where each row represents the coordinates (x, y, z) in a 3D space.
        density_value (np.array): density_value (np.array): A 1D NumPy array of shape (n_points,) containing the electron density values at the grid points.

    Returns:
        float: The D index, a scalar value representing the spatial separation between positive and negative density regions.
    """

    rho_p =  np.where(density_value < 0, 0, density_value)
    rho_m = np.where(density_value > 0, 0, density_value)

    tot_p = np.sum(rho_p)
    tot_m = np.sum(rho_m)

    x = density_grid[:,0]
    y = density_grid[:,1]
    z = density_grid[:,2]

    xp= np.dot(x,rho_p)/tot_p
    yp= np.dot(y,rho_p)/tot_p
    zp= np.dot(z,rho_p)/tot_p
    xm= np.dot(x,rho_m)/tot_m
    ym= np.dot(y,rho_m)/tot_m
    zm= np.dot(z,rho_m)/tot_m

    Dx = abs(xp - xm)
    Dy = abs(yp - ym)
    Dz = abs(zp - zm)

    D = (Dx**2 +Dy**2+ Dz**2)**(0.5)

    return(D)









