from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
import h5py
import pandas as pd

def density_projection(dust_xy, dust_xz, dust_yz):
    fig, axes = plt.subplots(2, 3, figsize=(15,10))

    # reading in the data
    with fits.open(dust_xy) as hdul:
        dust_density_xy = hdul[0].data
        header_xy = hdul[0].header

    with fits.open(dust_xz) as hdul:
        dust_density_xz = hdul[0].data
        header_xz = hdul[0].header

    with fits.open(dust_yz) as hdul:
        dust_density_yz = hdul[0].data
        header_yz = hdul[0].header

    # conversion to cgs units
    msun_to_g = u.Msun.to('g')
    au_to_cm = u.AU.to('cm')
    conv_factor = msun_to_g / (au_to_cm)**3
    dust_density_cgs_xy = dust_density_xy * conv_factor
    dust_density_cgs_xz = dust_density_xz * conv_factor
    dust_density_cgs_yz = dust_density_yz * conv_factor

    # data
    plot_data_row1 = [dust_density_xy, dust_density_xz, dust_density_yz]
    plot_data_row2 = [dust_density_cgs_xy, dust_density_cgs_xz, dust_density_cgs_yz]
    
    row_titles = [
        'Dust Density Projection (XY Plane)',
        'Dust Density Projection (XZ Plane)',
        'Dust Density Projection (YZ Plane)']
    
    row1_xlabel = ['X [pc]', 'X [pc]', 'Y [pc]']
    row1_ylabel = ['Y [pc]', 'Z [pc]', 'Z [pc]']
    row2_xlabel = ['X [pc]', 'X [pc]', 'Y [pc]']
    row2_ylabel = ['Y [pc]', 'Z [pc]', 'Z [pc]']

    # row 1 plots
    images_row1 = []
    for i, data in enumerate(plot_data_row1):
        im = axes[0, i].imshow(np.log10(data), cmap='viridis', origin='lower')
        images_row1.append(im)
        axes[0, i].set_xlabel(row1_xlabel[i])
        axes[0, i].set_ylabel(row1_ylabel[i])
        axes[0, i].set_title(row_titles[i])
        axes[0, i].set_xlim(480, 540)
        axes[0, i].set_ylim(480, 540)

    # single color bar for the first row
    fig.colorbar(images_row1[0], ax=axes[0, :].ravel().tolist(), label=r'$\log_{10}(\rho_{\rm dust})$ [Msun/AU$^3$]')
    
    # row 2 plots
    images_row2 = []
    for i, data in enumerate(plot_data_row2):
        im = axes[1, i].imshow(np.log10(data), cmap='viridis', origin='lower')
        images_row2.append(im)
        axes[1, i].set_xlabel(row2_xlabel[i])
        axes[1, i].set_ylabel(row2_ylabel[i])
        axes[1, i].set_title(row_titles[i])
        axes[1, i].set_xlim(480, 540)
        axes[1, i].set_ylim(480, 540)
    
    # single color bar for the second row
    fig.colorbar(images_row2[0], ax=axes[1, :].ravel().tolist(), label=r'$\log_{10}(\rho_{\rm dust})$ [g/cm$^3$]')

    #plt.tight_layout()
    plt.savefig('dust_density_dual_units_log.png', dpi=300)