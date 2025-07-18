import h5py
import numpy as np
from astropy import units as u
import time
import PTS9.utils as ut
import PTS9.simulation as sm
import PTS9.visual as vs
import PTS9.do

from snaputils import read_snapshot_info

# as of 6/6/25: adding in adaptiveness to the code. 

class SkirtSim:
    def __init__(self, snapshot, sim_type = 'sph', temp_file = 'template.ski'):
        '''
        Takes two file paths:
            snapshot: Path to the snapshot file. Default: 'snapshot_150.hdf5'.
            temp_file: Path to the ski template file for this project.

        Takes in a sim_type. Options are 'sph' and 'amr'.
        
        Attributes:
            self.snap_head: Dictionary to store snapshot header data.
            self.pt5: Stores source data from the snapshot header.
            self.pt0: Stores gas data from the snapshot header.
            self.src_data: Dictionary to store source data extracted from pt5.
            self.src_skirt: List to store source data formatted for SKIRT input.
            self.gas_data: Dictionary to store gas data extracted from pt0.
            self.gas_skirt: List to store gas data formatted for SKIRT input.
        '''
        self.snapshot = snapshot
        self.sim_type = sim_type
        self.snap_head = {} 
        self.pt5 = None
        self.pt0 = None
        self.src_data = {} 
        self.src_skirt = []
        self.gas_data = {}
        self.gas_skirt = []
        self.skitemp = temp_file

    def SnapInfo(self):
        '''
        Calls the external function to extract information from the snapshot header.

        This method now delegates the task of reading the snapshot file to
        the `read_snapshot_info` function from the `SnapUtils` module.
        It then assigns the returned header, PartType5, and PartType0 data
        to the corresponding attributes of the SkirtSim instance.
        '''
        self.snap_head, self.pt5, self.pt0 = read_snapshot_info(self.snapshot, self.sim_type)

        if not self.snap_head or self.pt5 is None or self.pt0 is None:
            print("Warning: Failed to load all snapshot information.")

            
    def computeTemperature(self, luminosity, star_radius):
        '''
        Computes the surface temperature of a star using its luminosity, radius, and the Stefan-Boltzmann Law.
        The Stefan-Boltzmann constant (sigma) used is 5.670374419e-5.
        Returns temperature in Kelvin.
        '''
        sigma = 5.670374419e-5
        T_sun = 5777 # k
        # AA adding this section in 7/10 to account for diff sim type units
        if self.sim_type == 'sph':
            return T_sun * (luminosity / star_radius**2)**0.25
        else:
            return (luminosity / (4 * np.pi * star_radius**2 * sigma))**0.25

    def SrcFile(self):
        '''
        Processes 'PartType5' (source) data from the snapshot.
        Calculates the temperature for each source using luminosity and star radius from 'self.pt5' and the 'computeTemperature' function.
        Populates 'self.src_data' with coordinates, smoothing length, radius, and temperature.
        Formats this data into a NumPy array 'self.src_skirt' for SKIRT input.
        Saves the processed data to an .hdf5 file with a custom header.
        '''
        pt5 = self.pt5

        # AA 7/15 adding in mass and radial cut and additional print statements
        masses = pt5['Masses']
        initial_num_sources = len(masses)
        print(f'Number of sources prior to any cuts: {initial_num_sources}')
        
        #mass_mask = np.where(masses >= 1)
        #print(f'Number of sources after the mass cut: {len(mass_mask[0])}')

        # mass cut section        
        luminosity_filt = pt5['Luminosity']#[mass_mask]
        star_radius_filt = pt5['Star Radius']#[mass_mask] # in solar radius for temp computation
        pt5['Temperature'] = self.computeTemperature(luminosity_filt, star_radius_filt)
        coords_filt = pt5['Coordinates']#[mass_mask]
        h_filt = pt5['BH_AccretionLength']#[mass_mask]

        # radial cut section
        r_extract = self.snap_head['Extraction Radius (pc)'] #* self.snap_head['Cloud Radius (pc)']
        center = self.snap_head['Center (pc)'] # Get the full center array here
        coords = coords_filt - center
        r_dist = np.linalg.norm(coords, axis=1)
        r_cut = r_dist < r_extract

        final_coords = coords[r_cut]
        final_h = h_filt[r_cut]
        final_R = star_radius_filt[r_cut]
        final_T = pt5['Temperature'][r_cut]
        #final_mass = mass_mask[r_cut]
        print(f'Number of sources after the radial cut: {len(final_T)}')
        
        #populating the dictionary
        # AA changing R units to km by multiplying by 6.95508e5 
        self.src_data = {
            'x(pc)': final_coords[:, 0],
            'y(pc)': final_coords[:, 1],
            'z(pc)': final_coords[:, 2],
            'h(pc)': final_h,
            'R(km)': final_R * 6.95508e5,
            'T(K)': final_T
        }
        #formatting for skirt
        self.src_skirt = np.column_stack([
            self.src_data['x(pc)'],
            self.src_data['y(pc)'],
            self.src_data['z(pc)'],
            self.src_data['h(pc)'],
            self.src_data['R(km)'],
            self.src_data['T(K)']
        ])

        #print and check
        print(f'Min x (pc) coord: {np.min(self.src_data["x(pc)"]):.2e}, Max x (pc) coord: {np.max(self.src_data["x(pc)"]):.2e}')
        print(f'Min y (pc) coord: {np.min(self.src_data["y(pc)"]):.2e}, Max y (pc) coord: {np.max(self.src_data["y(pc)"]):.2e}')
        print(f'Min z (pc) coord: {np.min(self.src_data["z(pc)"]):.2e}, Max z (pc) coord: {np.max(self.src_data["z(pc)"]):.2e}')
        print(f'Min smoothing length (pc): {np.min(self.src_data["h(pc)"]):.2e}, Max smoothing length (pc): {np.max(self.src_data["h(pc)"]):.2e}')
        print(f'Min radius (km): {np.min(self.src_data["R(km)"]):.2e}, Max radius (km): {np.max(self.src_data["R(km)"]):.2e}')
        print(f'Min temp (K): {np.min(self.src_data["T(K)"]):.2e}, Max temp (K): {np.max(self.src_data["T(K)"]):.2e}')
        
        #write and save the src file
        header = (
            "# x(pc) y(pc) z(pc) h(pc) R(km) T(K)"
        )
        filename = self.snapshot.replace('.hdf5', '_src.txt')
        np.savetxt(filename, self.src_skirt, fmt='%.6e', delimiter=' ', header=header, comments='')
        
        return filename


    def GasFile(self):
        '''
        Processes 'PartType0' (gas) data from the snapshot.
        Applies a radial cut based on the 'extraction radius' to filter particles.
        
        Before applying the cut:
            Centers coordinates by subtracting the snapshot center.
            Calculates the radial distance of each particle from the center.
            Keeps only particles within the extraction radius.
        
        Populates 'self.gas_data' with coordinates, smoothing length, masses, and temperatures.
        Formats this data into a NumPy array 'self.gas_skirt' for SKIRT input.
        Saves the processed data to an .hdf5 file with a custom header.
        '''
        pt0 = self.pt0

        # NEW as of 6/3: applying temperature cuts
        # anything eq or above 1500 K is assumed to have no dust
        # AA 7/10 removing temp cut
        #temp_threshold = 1500
        #temp_mask = pt0['Temperature'] < temp_threshold

        coords_filt = pt0['Coordinates']#[temp_mask]
        h_filt = pt0['SmoothingLength']#[temp_mask]
        masses_filt = pt0['Masses']#[temp_mask]
        print(f'Min original gas mass (Msun): {np.min(masses_filt):2e}, Max original gas mass (Msun): {np.max(masses_filt):2e}')
        temp_filt = pt0['Temperature']#[temp_mask]

        #get header data and apply the radial cut
        r_extract = self.snap_head['Extraction Radius (pc)'] #* self.snap_head['Cloud Radius (pc)']
        center = self.snap_head['Center (pc)'] # Get the full center array here
        print(f'Center is: {center}')

        coords = coords_filt - center
        r_dist = np.linalg.norm(coords, axis=1)
        r_cut = r_dist < r_extract

        final_coords = coords[r_cut]
        final_h = h_filt[r_cut]
        final_m_gas = masses_filt[r_cut]
        print(f'Min gas mass AFTER radial cut (Msun): {np.min(final_m_gas):2e}, Max gas mass AFTER radial cut (Msun): {np.max(final_m_gas):2e}')
        final_temp = temp_filt[r_cut]

        # NEW as of 6/3: applying gas to dust ratio
        # gas to dust ratio of 0.01
        gas_2_dust = 0.01
        final_m_dust = final_m_gas * gas_2_dust
        
        self.gas_data = {
            'x(pc)': final_coords[:, 0],
            'y(pc)': final_coords[:, 1],
            'z(pc)': final_coords[:, 2],
            'h(pc)': final_h,
            'M(Msun)': final_m_dust,
            'T(K)': final_temp
        }
        #formatting for skirt
        self.gas_skirt = np.column_stack([
            self.gas_data['x(pc)'],
            self.gas_data['y(pc)'],
            self.gas_data['z(pc)'],
            self.gas_data['h(pc)'],
            self.gas_data['M(Msun)'],
            self.gas_data['T(K)']
        ])

        #print and check
        print(f'Min x (pc) coord: {np.min(self.gas_data["x(pc)"]):.2e}, Max x (pc) coord: {np.max(self.gas_data["x(pc)"]):.2e}')
        print(f'Min y (pc) coord: {np.min(self.gas_data["y(pc)"]):.2e}, Max y (pc) coord: {np.max(self.gas_data["y(pc)"]):.2e}')
        print(f'Min z (pc) coord: {np.min(self.gas_data["z(pc)"]):.2e}, Max z (pc) coord: {np.max(self.gas_data["z(pc)"]):.2e}')
        print(f'Min smoothing length (pc): {np.min(self.gas_data["h(pc)"]):.2e}, Max smoothing length (pc): {np.max(self.gas_data["h(pc)"]):.2e}')
        print(f'Min dust mass (Msun): {np.min(self.gas_data["M(Msun)"]):.2e}, Max dust mass (Msun): {np.max(self.gas_data["M(Msun)"]):.2e}')
        print(f'Min temp (K): {np.min(self.gas_data["T(K)"]):.2e}, Max temp (K): {np.max(self.gas_data["T(K)"]):.2e}')
        
        #write and save the gas file
        header = (
            "# column 1: x(pc)\n"
            "# column 2: y(pc)\n"
            "# column 3: z(pc)\n"
            "# column 4: h(pc)\n"
            "# column 5: M(Msun)\n"
            "# column 6: T(K)"
        )
        filename = self.snapshot.replace('.hdf5', '_gas.txt')
        np.savetxt(filename, self.gas_skirt, fmt='%.6e', delimiter=' ', header=header, comments='')
        
        return filename

    def SkiFile(self, gasFile, srcFile):
        '''
        Generates the SKIRT configuration file (.ski) by updating a template file with simulation parameters and data filenames.
        
        Args:
            gasFile (str): Filename of the processed gas data for SKIRT configuration.
            srcFile (str): Filename of the processed source data for SKIRT configuration.
        
        Reads the template .ski file and replaces placeholders.
        The updated .ski file is then written.
        '''
        header = self.snap_head
        boxsize = header['BoxSize (pc)']
        center = header['Center (pc)']
    
        #setting the output SKIRT file name based on the HDF5 file
        filename = self.snapshot.replace('.hdf5', '')
    
        #read the SKIRT template file
        skitemp = "template.ski" 
        with open(skitemp, 'r') as f:
            filedata = f.read()
    
        filedata = filedata.replace('"stars.txt"', f'"{srcFile}"')
        filedata = filedata.replace('"gas.txt"', f'"{gasFile}"')
    
    
        #replacing min and max bounds for X, Y, and Z based on the simulation box size
        for axis in ['X', 'Y', 'Z']:
            filedata = filedata.replace(f'min{axis}="-{axis}max pc"', f'min{axis}="-{center[0]} pc"')
            filedata = filedata.replace(f'max{axis}="{axis}max pc"', f'max{axis}="{center[0]} pc"')
    
        #write the new SKIRT configuration file
        skifile = filename + '.ski'
        with open(skifile, 'w') as f:
            f.write(filedata)
    
        return skifile

    def SimRun(self):
        '''
        Manages the entire SKIRT simulation process.
        Calls methods to read snapshot information, create gas and source data files, generate the .ski configuration file, and execute the SKIRT simulation.
        Includes a timing method to measure total simulation runtime.
        
        Steps:
            1. Attempts to open and read the snapshot file using `self.SnapInfo()`.
            2. Creates the SKIRT-formatted source file using `self.SrcFile()`.
            3. Creates the SKIRT-formatted gas file using `self.GasFile()`.
            4. Generates the .ski file using `self.SkiFile()`, providing paths for the gas and source files.
            5. Executes the SKIRT simulation using the generated .ski file.
        
        Returns:
            gas_data (dict): The processed gas data dictionary.
            src_data (dict): The processed source data dictionary.
            sim: The SKIRT simulation result.
        '''
        start_time = time.time() #start timing
        print('Starting timing for simulation.')

        try:
            print(f'Attempting to open snapshot file: {self.snapshot}')
            self.SnapInfo()
            print('File was opened successfully.')
            # get src file
            srcFile = self.SrcFile()
            # get gas file
            gasFile = self.GasFile()
            print('SKIRT source and gas files created')
            #create the ski file from the template
            skiFile = self.SkiFile(gasFile, srcFile) 
            print('SKIRT .ski file created.')
            #execute simulation
            skirt = sm.Skirt()
            sim = skirt.execute(skiFile, console='brief')
            print('SKIRT sim executed.')

            end_time = time.time()  # End timing
            print("Ending timing.")
            elapsed_time = end_time - start_time
            print(f"Execution time: {elapsed_time:.2e} seconds")
                
            return self.gas_data, self.src_data, sim

        except FileNotFoundError:
            print(f'Error: File {self.snapshot} not found.')
            return None, None, None
#process the snapshot file
if __name__ == "__main__":
    snapshot_file = 'snapshot_150.hdf5' 
    sim = SkirtSim(snapshot_file)
    gas_data, source_data, sim_result = sim.SimRun()