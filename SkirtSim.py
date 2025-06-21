import h5py
import numpy as np
from astropy import units as u
import time
import PTS9.utils as ut
import PTS9.simulation as sm
import PTS9.visual as vs
import PTS9.do
import yt

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
        Extracts information from the snapshot header.
        
        Reads header attributes to determine 'BoxSize', used for calculating 'Center' and 'Cloud Radius'.
        Also extracts snapshot time, coordinates, and other properties for 'PartType5' (sources) and 'PartType0' (gas).
        
        Populates the following dictionaries with simulation properties:
            self.snap_head: Contains header information such as 'BoxSize (pc)', 'Center (pc)',
                             'Cloud Radius (pc)', 'Extraction Radius (pc)', and 'Snapshot Time (yr)'.
            self.pt5: Contains data for 'PartType5', including 'Coordinates', 'BH_AccretionLength',
                       'Star Radius', and 'Luminosity'.
            self.pt0: Contains data for 'PartType0', including 'Coordinates', 'SmoothingLength',
                       'Masses', and 'Temperature'.
        
        Note: 'r_extract' is defined as 0.10 pc for testing purposes. It is typically set to 0.25 pc.
        '''
        r_extract = 0.10
        if self.sim_type == 'sph':
            with h5py.File(self.snapshot, 'r') as f:
                header = f['Header'].attrs
                self.snap_head = {
                    'BoxSize (pc)': header['BoxSize'], # this is not in pc actually but code length based off of yt. itll be pc if we multiply by 1000
                    'Center (pc)': np.full(3, (header['BoxSize']) / 2),
                    'Cloud Radius (pc)': (header['BoxSize']) / 10,
                    'Extraction Radius (pc)': r_extract,
                    'Snapshot Time (yr)': header['Time'] * (u.pc / (u.m / u.s)).to('yr')
                }
                self.pt5 = {
                    'Coordinates': f['PartType5']['Coordinates'][:],
                    'BH_AccretionLength': f['PartType5']['BH_AccretionLength'][:],
                    'Star Radius': f['PartType5']['ProtoStellarRadius_inSolar'][:],
                    'Luminosity': f['PartType5']['StarLuminosity_Solar'][:]
                }
                self.pt0 = {
                    'Coordinates': f['PartType0']['Coordinates'][:],
                    'SmoothingLength': f['PartType0']['SmoothingLength'][:],
                    'Masses': f['PartType0']['Masses'][:],
                    'Temperature': f['PartType0']['Temperature'][:]
                }
      #  elif self.sim_type == 'amr':
       #     ds = yt.load(self.snapshot)
        #    self.snap_head = {
         #       'BoxSize (pc)': ,
          #      'Center (pc)': ,
           #     'Cloud Radius (pc)': ,
            #    'Extraction Radius (pc)': r_extract,
             #   'Snapshot Time (yr)':
            #}
            
    def computeTemperature(self, luminosity, star_radius):
        '''
        Computes the surface temperature of a star using its luminosity, radius, and the Stefan-Boltzmann Law.
        The Stefan-Boltzmann constant (sigma) used is 5.670374419e-5.
        Returns temperature in Kelvin.
        '''
        sigma = 5.670374419e-5 
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
        luminosity = pt5['Luminosity']
        star_radius = pt5['Star Radius']
        pt5['Temperature'] = self.computeTemperature(luminosity, star_radius)
        #populating the dictionary
        self.src_data = {
            'x (pc)': pt5['Coordinates'][:, 0],
            'y (pc)': pt5['Coordinates'][:, 1],
            'z (pc)': pt5['Coordinates'][:, 2],
            'h (pc)': pt5['BH_AccretionLength'][:],
            'r (pc)': pt5['Star Radius'][:],
            'T (K)': pt5['Temperature'][:]
        }
        #formatting for skirt
        self.src_skirt = np.column_stack([
            self.src_data['x (pc)'],
            self.src_data['y (pc)'],
            self.src_data['z (pc)'],
            self.src_data['h (pc)'],
            self.src_data['r (pc)'],
            self.src_data['T (K)']
        ])
        #write and save the src file
        header = (
            "# x (pc) y (pc) z (pc) smoothingLength (pc) radius (pc) temperature (K)"
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
        # anything eq or above 15000 K is assumed to have no dust
        temp_threshold = 15000
        temp_mask = pt0['Temperature'] < temp_threshold

        coords_filt = pt0['Coordinates'][temp_mask]
        h_filt = pt0['SmoothingLength'][temp_mask]
        masses_filt = pt0['Masses'][temp_mask]
        temp_filt = pt0['Temperature'][temp_mask]

        #get header data and apply the radial cut
        r_extract = self.snap_head['Extraction Radius (pc)']
        center = self.snap_head['Center (pc)'][0]

        coords = coords_filt - center
        r_dist = np.linalg.norm(coords, axis=1)
        r_cut = r_dist < r_extract

        final_coords = coords[r_cut]
        final_h = h_filt[r_cut]
        final_m_gas = masses_filt[r_cut]
        final_temp = temp_filt[r_cut]

        # NEW as of 6/3: applying gas to dust ratio
        # gas to dust ratio of 0.01
        gas_2_dust = 0.01
        final_m_dust = final_m_gas * gas_2_dust
        
        self.gas_data = {
            'x (pc)': final_coords[:, 0],
            'y (pc)': final_coords[:, 1],
            'z (pc)': final_coords[:, 2],
            'h (pc)': final_h,
            'M (Msun)': final_m_dust,
            'T (K)': final_temp
        }
        #formatting for skirt
        self.gas_skirt = np.column_stack([
            self.gas_data['x (pc)'],
            self.gas_data['y (pc)'],
            self.gas_data['z (pc)'],
            self.gas_data['h (pc)'],
            self.gas_data['M (Msun)'],
            self.gas_data['T (K)']
        ])
        #write and save the gas file
        header = (
            "# column 1: x position (pc)\n"
            "# column 2: y position (pc)\n"
            "# column 3: z position (pc)\n"
            "# column 4: smoothing length (pc)\n"
            "# column 5: dust mass (Msun)\n"
            "# column 6: temperature (K)"
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
            print(f"Execution time: {elapsed_time:.2f} seconds")
                
            return self.gas_data, self.src_data, sim

        except FileNotFoundError:
            print(f'Error: File {self.snapshot} not found.')
            return None, None, None
#process the snapshot file
if __name__ == "__main__":
    snapshot_file = 'snapshot_150.hdf5' 
    sim = SkirtSim(snapshot_file)
    gas_data, source_data, sim_result = sim.SimRun()