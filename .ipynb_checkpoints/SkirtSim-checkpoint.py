import h5py
import numpy as np
from astropy import units as u
import time
import PTS9.utils as ut
import PTS9.simulation as sm
import PTS9.visual as vs
import PTS9.do

class SkirtSim:
    def __init__(self, snapshot, temp_file = 'template.ski'):
        '''
        This function initializes what will be needed for the entire class. It takes in two files: the snapshot and the ski template files. 
            self.snapshot: The snapshot file. The default used for this code is 'snapshot_150.hdf5'. 
            self.snap_head: This is an empty dictionary that will eventually hold all the snapshot header data.
            self.pt5: This will eventually hold all the source data from the snapshot header.
            self.pt0: This will eventually hold all the gas data from the snapshot header.
            self.src_data: This is an empty dictionary that will eventually hold the source data extracted from the snapshot file, specifically from pt5.
            self.src_skirt: This is an empty list that will eventually hold the source data and will be formatted for SKIRT input.
            self.gas_data: This is an empty dictionary that will eventually hold the gas data extracted from the snapshot file, specifically from pt0.
            self.gas_skirt: This is an empty list that will eventually hold the gas data and will be formatted for SKIRT input.
            self.skitemp: The ski template file. We use the template ski file that was created for this project. 
        '''
        self.snapshot = snapshot
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
        This function extracts the information from the snapshot header. It reads in the header attributes to get the boxsize which is used to calculate the center and the cloud radius. It also extracts the snapshot time, the coordinates, and other properties for the sources (PartType5) and the gas (PartType0). 

        The initialized dictionaries (self.snap_head, self.pt5, self.pt0) are populated to contain all the necessary properties for the simulation run:
            self.snap_head: A dictionary containing header information such as
                           'BoxSize (pc)', 'Center (pc)', 'Cloud Radius (pc)',
                           'Extraction Radius (pc)', and 'Snapshot Time (yr)'.
                           
            self.pt5: A dictionary containing data for PartType5, including
                      'Coordinates', 'BH_AccretionLength', 'Star Radius', and
                      'Luminosity'.
                      
            self.pt0: A dictionary containing data for PartType0, including
                      'Coordinates', 'SmoothingLength', 'Masses', and 'Temperature'.            

        Here we define r_extract to be 0.10 pc. ** Usually, this would be set to 0.25 pc, but we are currently using a smaller set of data for testing. **
        '''
        r_extract = 0.10
        
        with h5py.File(self.snapshot, 'r') as f:
            header = f['Header'].attrs
            self.snap_head = {
                'BoxSize (pc)': header['BoxSize'], 
                'Center (pc)': np.full(3, header['BoxSize'] / 2),
                'Cloud Radius (pc)': header['BoxSize'] / 10,
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
            
    def computeTemperature(self, luminosity, star_radius):
        '''
        This function computes the surface temperature using the luminosity and the radius of the star and the Stefan-Boltzmann law. This function is used to get the temperature for the source file.
        
        The Stefan-Boltzmann constant used in the calculation is sigma = 5.670374419e-5. The returned temperature will be in Kelvin. 
        '''
        sigma = 5.670374419e-5 
        return (luminosity / (4 * np.pi * star_radius**2 * sigma))**0.25

    def SrcFile(self):
        '''
        This function processes the source (PartType5) data from the snapshot, computes the temperature for each source, formats the data for SKIRT, and saves it to a text file. The temperature is calculated by taking the luminosity and star radius from 'self.pt5' and using the 'computeTemperature' function. The dictionary 'self.src_data' is then filled with the coordinates, the smoothing length, the radius, and the temperature. The dictionary is then formatted into a numpy array 'self.src_skirt' as this is better for SKIRT input. The file is then saved as an .hdf5 file with a custom header. 
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
        This function processes the gas (PartType0) data from the snapshot, applies the radial cut (which is from the 'extraction radius'), formats the data for SKIRT, and save it to a text file. 

        Before applying the radial cut, we center the coordinates by subtracting off the snapshot center and calculate the radial distance of each particle from the center. Then we apply the cut to only keep particles within the extration radius. 

        The dictionary 'self.gas_data' is then filled with the coordinates, smoothing length, masses, and temperatures. The dictionary is then formatted into a numpy array 'self.gas_skirt' as this is better for SKIRT input. The file is then saved as an .hdf5 file with a custom header. 
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
        This function generates the SKIRT configuration file (`.ski`) by changing the template file with simulation parameters and data filenames.

        Args listed:
            gasFile: The filename of the processed gas data to be used in the SKIRT configuration.
            srcFile: The filename of the processed source data to be used in the SKIRT configuration.

        This function reads in the template ski file and replaces any placeholders. For example, 'stars.txt' is replaced with the actual source data file. Another example is the min and max values of the axes are replaced with the actual box dimensions. After all the replacements, the new ski file is written.
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
        This function does the entire SKIRT simulation process. It calls the methods to read snapshot information, create gas and source files, generate the ski file, and execute the SKIRT simulation. This function also includes a timing method to see how long the simulation takes to fully run. 

        The following steps are done in the function:
        1. Attempt to open and read the snapshot file using `self.SnapInfo()`.
        2. Create the SKIRT-formatted source file using `self.SrcFile()`.
        3. Create the SKIRT-formatted gas file using `self.GasFile()`.
        4. Generate the ski file using `self.SkiFile()`, using the paths of the gas and source files.
        5. Execute the SKIRT simulation using the ski file.

        This function returns:
            - gas_data: The processed gas data dictionary
            - src_data: The processed source data dictionary
            - sim: The SKIRT simulation result
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