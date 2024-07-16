import rich_click as click 
from pathlib import Path
from sys import exit
from subprocess import run
import numpy as np

import utils
import read_file
import clustering_structure
import data_analysis as da



__version__ = '1.1.0'




@click.command()
@click.rich_config(help_config=click.RichHelpConfiguration(use_markdown=True, width=60))
@click.option('-f', '--filenameTD','fName', required=True, type = str, help = 'Transition density file to analyze')
@click.option('-f0', '--filenameGround','fName0', required=False, type = str, help = 'Ground state charge density file')
@click.option('-fd', '--filenameDD','fNameDD', required=False, type = str, help = 'Difference density file to analyze')
@click.option('-r', '--remote', default = None, type= str, help = 'Download file from server. Example: username@remote_server:/path')

@click.option('-m', '--manual', default = [], required=None, multiple=True, type=int, help='For manual clusterization')

@click.option('-v', '--verbosity', default=False,  is_flag=True,  required=None, type=bool, help="Print more information")
@click.option('-p', '--plot', default=False, is_flag=True, required=None, type=bool, help="Plot the clusterization results to check them")

def action(fName:str, fName0:str, fNameDD: str, remote:str, manual, plot, verbosity:bool ):  
    '''This tool uses a transition density file to analyze and classify excitations for hybrid systems composed of a metal nanoparticle and one or more molecules.
      You can also analyze the difference density file in order to distinguish between LE or CT states. 
      The code automatically recognizes the subsystems. To check the clusterization, use the flag -c.
    '''
    
    verbosity_flag = verbosity

    ###################################################################################################
    ###############################      Ground state density file      ###############################
    ###################################################################################################
    

    if fName0:
        fName0 = Path(fName0)
        if remote is None:
            if not fName0.exists():
                print(f"The file {fName0} does not exist")
                exit(1)
            if not fName0.suffix == '.cube' and not fName0.suffix == '.cub':
                print("The file name must have the extension .cube or .cub")
                exit(1)
        else:
            if not remote.endswith('/'):
                remote=f'{remote}/'
            a=run(f'scp {remote}{fName0} ./', text=True)
            if a.returncode == 1:
                print(f"The file {fName0} does not exist")
                exit(1)

        # read the file 
        parse_data = read_file.parse_cube_file((f'{fName0}'))
        atomic_info = read_file.read_coordinates(parse_data)  

        atom_type = atomic_info[:, 0]
        atomic_coord = atomic_info[:, 1:4]
        radii = atomic_info[:, 4]
        charge = atomic_info[:, 5]


        density_info, vol = read_file.read_density(parse_data) 
    
        density_coord = density_info[:, :3]
        density_per_point = density_info[:, -1]

        weight = da.optimization_radius(atom_type, atomic_coord, charge , density_coord , density_per_point, vol, radii)
        print('Weight:', weight)
        
        if verbosity_flag:
            print('Weight:', weight)
        
    else:
        weight = 0.75
        print('The weight could be inaccurate.')
    


    ###################################################################################################
    #######################################         TD file     #######################################
    ###################################################################################################


    fName=Path(fName)
    if remote is None:
        if not fName.exists():
            print(f"The file {fName} does not exist")
            exit(1)
        if not fName.suffix == '.cube' and not fName.suffix == '.cub':
            print("The file name must have the extension .cube or .cub")
            exit(1)
    else:
        if not remote.endswith('/'):
            remote=f'{remote}/'
        a=run(f'scp {remote}{fName} ./', text=True)
        if a.returncode == 1:
            print(f"The file {fName} does not exist")
            exit(1)

    # read the file 
    parse_data = read_file.parse_cube_file((f'{fName}'))
    atomic_info = read_file.read_coordinates(parse_data)  

    atom_type = atomic_info[:, 0]
    atomic_coord = atomic_info[:, 1:4]
    radii = atomic_info[:, 4]
    charge = atomic_info[:, 5]

    
    # clusterization
    if not manual:
        cluster_model = clustering_structure.dbscan_model(atomic_coord) 
        clustered_data = clustering_structure.separate_data_by_labels(atomic_info, cluster_model.labels_) 
        #print(f"This is the clustered dataset from clustering model:{clustered_data}")
    else:
        manual_list = clustering_structure.split_function_manual_mode(manual, atomic_info, verbosity)
        #print(f"This is the clustered dataset from manual:{manual_list}")
    
    
    # index calculation
    density_info, vol = read_file.read_density(parse_data) 
    
    density_coord = density_info[:, :3]
    density_per_point = density_info[:, -1]

    dict_charge_system = {}
    if not manual: 
        for key in clustered_data:
            atomic_coord = np.array(clustered_data[key])[:, 1:4]
            radii = np.array(clustered_data[key])[:, 4]
            
            charge_per_atom, charge_per_subst = da.charge4atom(atomic_coord, density_coord, density_per_point, vol, radii, weight, verbosity_flag)
            dict_charge_system[key]= charge_per_subst
            #print('Charge on', key, ':', round(charge_per_subst,2) )

    else:
        for key in manual_list:
            atomic_coord = np.array(manual_list[key])[:, 1:4]
            radii = np.array(manual_list[key])[:, 4]
            
            charge_per_atom, charge_per_subst = da.charge4atom(atomic_coord, density_coord, density_per_point, vol, radii, weight, verbosity_flag)
            dict_charge_system[key]= charge_per_subst
            #print('Charge on', key,':', round(charge_per_subst,2) )

    if verbosity_flag:
        print('Charge per atom:\n', charge_per_atom)
        print('Charge per subsystem:\n', charge_per_subst)

    
    index_dict = {}
    for key in dict_charge_system:
        if key != 'metal_cluster':
            hybridiz_index = da.hybridization_index(dict_charge_system['metal_cluster'], dict_charge_system[f'{key}'])
            index_dict[key]= hybridiz_index
            
  
    for key, value in index_dict.items():
        if key != 'metal_cluster':
            print(f"{key}/metal_cluster, HI value: {value}.")
            if value > 1.5:
                print("The excitation is molecular like.")
            elif value < 0.1:
                print("The excitation is metal like.")
            else:
                print("The excitation is hybrid.")
    
 

    
    # plot the system, highlighting the different parts
    if plot:
        if not manual:
            clustering_structure.plot_coordinates(clustered_data)
            clustering_structure.plot_3d_dataset(clustered_data)
           
        else:
            clustering_structure.plot_coordinates(manual_list)
            clustering_structure.plot_3d_dataset(manual_list)





    ###################################################################################################
    #######################################         DD file     #######################################
    ###################################################################################################    
    if fNameDD:
        fNameDD=Path(fNameDD)
        if remote is None:
            if not fNameDD.exists():
                print(f"The file {fNameDD} does not exist")
                exit(1)
            if not fNameDD.suffix == '.cube' and not fNameDD.suffix == '.cub':
                print("The file name must have the extension .cube o .cub")
                exit(1)
            else: 
                parse_data = read_file.parse_cube_file((f'{fNameDD}'))
                density_info, _ = read_file.read_density(parse_data) 

                density_coord = density_info[:, :3]
                density_per_point = density_info[:, -1]
                
                Dindex = da.ct_finder(density_coord, density_per_point)
                print('D index', round(Dindex,2), 'angstrom')

        else:
            if not remote.endswith('/'):
                remote=f'{remote}/'
            a=run(f'scp {remote}{fNameDD} ./', text=True)   
            if a.returncode == 1:
                print(f"The file {fNameDD} does not exist")
                exit(1)
            else: 
                parse_data = read_file.parse_cube_file((f'{fNameDD}'))
                density_info, _ = read_file.read_density(parse_data) 

                density_coord = density_info[:, :3]
                density_per_point = density_info[:, -1]

                Dindex = da.ct_finder(density_coord, density_per_point)
                print('D index', round(Dindex,2), 'angstrom')


    

######################################################################################################

if __name__ == '__main__':
    action()
    