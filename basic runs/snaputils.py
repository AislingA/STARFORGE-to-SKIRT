import h5py
import numpy as np
from astropy import units as u

def read_snapshot_info(snapshot_path, sim_type='sph'):
    """
    Extracts information from the snapshot header.

    Reads header attributes to determine 'BoxSize', used for calculating 'Center' and 'Cloud Radius'.
    Also extracts snapshot time, coordinates, and other properties for 'PartType5' (sources) and 'PartType0' (gas).

    Args:
        snapshot_path (str): Path to the snapshot file (e.g., 'snapshot_150.hdf5').
        sim_type (str): Type of simulation, either 'sph' or 'amr'. Currently, only 'sph' is implemented.

    Returns:
        tuple: A tuple containing three dictionaries:
            - snap_head (dict): Contains header information such as 'BoxSize (pc)', 'Center (pc)',
                                'Cloud Radius (pc)', 'Extraction Radius (pc)', and 'Snapshot Time (yr)'.
            - pt5 (dict): Contains data for 'PartType5', including 'Coordinates', 'BH_AccretionLength',
                          'Star Radius', and 'Luminosity'.
            - pt0 (dict): Contains data for 'PartType0', including 'Coordinates', 'SmoothingLength',
                          'Masses', and 'Temperature'.

    Note: 'r_extract' is defined as 0.10 pc for testing purposes. It is typically set to 0.25 pc.
    """
    r_extract = 0.10 # Extraction radius in pc
    snap_head = {}
    pt5 = None
    pt0 = None

    if sim_type == 'sph':
        try:
            with h5py.File(snapshot_path, 'r') as f:
                header = f['Header'].attrs
                snap_head = {
                    'BoxSize (pc)': header['BoxSize'],
                    'Center (pc)': np.full(3, (header['BoxSize']) / 2),
                    'Cloud Radius (pc)': (header['BoxSize']) / 10,
                    'Extraction Radius (pc)': r_extract, # instead of r_extract, changing this to the cloud radius so those will be equal to each other
                    'Snapshot Time (yr)': header['Time'] * (u.pc / (u.m / u.s)).to('yr')
                }
                pt5 = {
                    'Coordinates': f['PartType5']['Coordinates'][:],
                    'BH_AccretionLength': f['PartType5']['BH_AccretionLength'][:],
                    'Star Radius': f['PartType5']['ProtoStellarRadius_inSolar'][:],
                    'Luminosity': f['PartType5']['StarLuminosity_Solar'][:]
                }
                pt0 = {
                    'Coordinates': f['PartType0']['Coordinates'][:],
                    'SmoothingLength': f['PartType0']['SmoothingLength'][:],
                    'Masses': f['PartType0']['Masses'][:],
                    'Temperature': f['PartType0']['Temperature'][:]
                }
        except KeyError as e:
            print(f"Error reading HDF5 file. Missing key: {e}. Ensure all expected datasets are present.")
            snap_head, pt5, pt0 = {}, None, None
        except FileNotFoundError:
            print(f"Error: Snapshot file not found at {snapshot_path}.")
            snap_head, pt5, pt0 = {}, None, None
    elif sim_type == 'amr':
        # Placeholder for AMR implementation
        print("AMR simulation type not yet fully implemented for header reading.")
        # You would load with yt here and populate the dictionaries similarly
        # ds = yt.load(snapshot_path)
        # snap_head = {
        #    'BoxSize (pc)': ds.domain_width.to('pc').value, # Example
        #    'Center (pc)': ds.domain_center.to('pc').value, # Example
        #    'Cloud Radius (pc)': ds.domain_width[0].to('pc').value / 10, # Example
        #    'Extraction Radius (pc)': r_extract,
        #    'Snapshot Time (yr)': ds.current_time.to('yr').value # Example
        # }
        # pt5 = ... # Extract data for PartType5 if applicable in AMR
        # pt0 = ... # Extract data for PartType0 if applicable in AMR
    else:
        print(f"Unsupported simulation type: {sim_type}. Only 'sph' and 'amr' are supported.")

    return snap_head, pt5, pt0