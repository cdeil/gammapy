"""Spectral plotting with gammapy.spectrum.flux_point
"""
import numpy as np
from astropy.table import Table
from gammapy.spectrum.flux_point import compute_differential_flux_points, _energy_lafferty_power_law
from gammapy.spectrum.powerlaw import power_law_eval, power_law_integral_flux
import matplotlib.pyplot as plt

SPECTRAL_INDEX = 4

def my_spectrum(x):
    E_1 = 1
    E_2 = 10
    E_3 = 100
    E_4 = 1000
    E_5 = 10000
    g1 = -1
    g2 = -2
    g3 = -3
    g4 = -4
    g5 = -5
    im = np.select([x <= E_1, x <= E_2, x <= E_3, x <= E_4, x <= E_5, x > E_5],
                   [(x / E_1) ** g1, 1e-2 * (x / E_2) ** g2,
                    1e-5 * (x / E_3) ** g3, 1e-9 * (x / E_4) ** g4,
                    1e-14 * (x / E_5) ** g5, 0])
    return im
  
    
def get_flux_tables(table, y_method, function, spectral_index):
    table1 = table.copy()
    lafferty_flux = compute_differential_flux_points(table1, 'lafferty', y_method,
                                                     function, spectral_index)
    table2 = table.copy()
    log_flux = compute_differential_flux_points(table2, 'log_center', y_method,
                                                function, spectral_index)
    return lafferty_flux, log_flux


def plot_flux_points(table, x, y, function, energy_min, energy_max, y_method):
    f, axarr = plt.subplots(2, sharex=True)
    lafferty_flux, log_flux = get_flux_tables(table, y_method, function, SPECTRAL_INDEX)

    axarr[0].loglog(x, (x ** 2) * y)
    axarr[0].loglog(lafferty_flux['ENERGY'],
                    ((lafferty_flux['ENERGY'] ** 2) * lafferty_flux['DIFF_FLUX']),
                    marker='D', color='k', linewidth=0, ms=5, label='Lafferty Method')
    residuals_lafferty = (lafferty_flux['DIFF_FLUX']
                          - function(lafferty_flux['ENERGY'])) / function(lafferty_flux['ENERGY']) * 100
    axarr[0].loglog(log_flux['ENERGY'],
                    (log_flux['ENERGY'] ** 2) * log_flux['DIFF_FLUX'],
                    marker='D', color='r', linewidth=0, ms=5, label='Log Center Method')
    axarr[0].legend(loc='lower left', fontsize=10)

    residuals_log = (log_flux['DIFF_FLUX'] - 
                     function(log_flux['ENERGY'])) / function(log_flux['ENERGY']) * 100
    axarr[1].semilogx(lafferty_flux['ENERGY'], residuals_lafferty, marker='D',
                      color='k', linewidth=0, ms=5)
    axarr[1].semilogx(log_flux['ENERGY'], residuals_log, marker='D',
                      color='r', linewidth=0, ms=5)
    indices = np.arange(len(energy_min))
    for index in indices:
        axarr[0].axvline(energy_min[index], 0, 1e6, color='k', linestyle=':')
        axarr[1].axvline(energy_min[index], 0, 1e6, color='k', linestyle=':')
    axarr[1].axhline(0, 0, 10, color='k')
    axarr[0].set_ylabel('E^2 * Differential Flux')
    axarr[1].set_ylabel('Residuals/%')
    axarr[1].set_xlabel('Energy')
    axarr[0].set_xlim([0.1, 10000])
    axarr[0].set_ylim([1e-6, 1e1])
    return plt


def plot_plaw():
    # Define the function
    x = np.arange(0.1, 100000, 0.1)
    spectral_model = my_spectrum(x)
    spectral_model_function = lambda x: my_spectrum(x)
    y = spectral_model
    # Set the x-bins
    energy_min = [0.1, 1, 10, 100, 1000]
    energy_max = [1, 10, 100, 1000, 10000]
    # Integral fluxes (determined by working backwards)
    int_flux = [2.30159284241, 0.8999901655, 0.0494999902499,
                0.000332999999032, 2.49974999996e-07]
    # Put data into table
    table = Table()
    table['ENERGY_MIN'] = energy_min
    table['ENERGY_MAX'] = energy_max
    table['INT_FLUX'] = int_flux

    plt = plot_flux_points(table, x, y, spectral_model_function,
                           energy_min, energy_max, 'power_law')
    plt.legend()
    return plt


def make_gamma_plot():
    energy_min = np.array([3])
    energy_max = np.array([1000])
    energies = np.array(_energy_lafferty_power_law(energy_min, energy_max, SPECTRAL_INDEX))
    diff_flux =  power_law_eval(energies, 1, SPECTRAL_INDEX, 1)
    int_flux = power_law_integral_flux(diff_flux, SPECTRAL_INDEX, energies, energy_min, energy_max)
    # Put data into table
    table = Table()
    table['ENERGY_MIN'] = energy_min
    table['ENERGY_MAX'] = energy_max
    table['INT_FLUX'] = int_flux
    
    spectral_indices = np.arange(1.1, 6, 0.01)
    for spectral_index in spectral_indices:
        lafferty_flux, log_flux = get_flux_tables(table, 'power_law', None, spectral_index)
        residuals_lafferty = ((lafferty_flux['DIFF_FLUX']
                              - diff_flux) / diff_flux) * 100
        plt.plot(spectral_index, residuals_lafferty, marker='D',
                     color='k', linewidth=0, ms=1)
        #residuals_log = ((log_flux['DIFF_FLUX']
        #                     - diff_flux) / diff_flux) * 100
        #plt.plot(spectral_index, residuals_log, marker='D',
        #             color='r', linewidth=0, ms=1)
    
   #indices = np.arange(len(energy_min))
   # for index in indices:
   #     plt.vlines(energy_min[index], -100, 800, color='k', linestyle=':')
    #plt.hlines(0, 0, 10, color='k')
    plt.ylabel('Residuals/%')
    plt.xlabel('Spectral Index')
    #plt.xlim([1, 10000])
    plt.show()
    
def make_x_plot():
    energy_min = np.array([3])
    energy_max = np.array([1000])
    energies = np.array(_energy_lafferty_power_law(energy_min, energy_max, SPECTRAL_INDEX))
    diff_flux =  power_law_eval(energies, 1, SPECTRAL_INDEX, 1)
    int_flux = power_law_integral_flux(diff_flux, SPECTRAL_INDEX, energies, energy_min, energy_max)
    # Put data into table
    table = Table()
    table['ENERGY_MIN'] = energy_min
    table['ENERGY_MAX'] = energy_max
    table['INT_FLUX'] = int_flux
    
    spectral_indices = np.arange(1.1, 6, 0.01)
    for spectral_index in spectral_indices:
        lafferty_flux, log_flux = get_flux_tables(table, 'power_law', None, spectral_index)
        residuals_lafferty = ((lafferty_flux['ENERGY']
                              - energy_min) / (energy_max-energy_min))
        plt.plot(spectral_index, residuals_lafferty, marker='D',
                     color='k', linewidth=0, ms=1)
        residuals_log = ((log_flux['ENERGY']
                              - energy_min) / (energy_max-energy_min))
        plt.plot(spectral_index, residuals_log, marker='D',
                     color='r', linewidth=0, ms=1)
    
   #indices = np.arange(len(energy_min))
   # for index in indices:
   #     plt.vlines(energy_min[index], -100, 800, color='k', linestyle=':')
    #plt.hlines(0, 0, 10, color='k')
    plt.ylabel('X position in bin')
    plt.xlabel('Spectral Index')
    #plt.xlim([1, 10000])
    plt.show()
    
def make_dex_plot():
    # TODO (in progress)
    energy_min = np.array([3])
    energy_max = np.array([1000])
    energies = np.array(_energy_lafferty_power_law(energy_min, energy_max, SPECTRAL_INDEX))
    diff_flux =  power_law_eval(energies, 1, SPECTRAL_INDEX, 1)
    int_flux = power_law_integral_flux(diff_flux, SPECTRAL_INDEX, energies, energy_min, energy_max)
    # Put data into table
    table = Table()
    table['ENERGY_MIN'] = energy_min
    table['ENERGY_MAX'] = energy_max
    table['INT_FLUX'] = int_flux
    

    lafferty_flux, log_flux = get_flux_tables(table, 'power_law', None, spectral_index)
    residuals_lafferty = ((lafferty_flux['DIFF_FLUX']
                            - diff_flux) / diff_flux) * 100
    plt.plot(spectral_index, residuals_lafferty, marker='D',
                    color='k', linewidth=0, ms=1)
        #residuals_log = ((log_flux['DIFF_FLUX']
        #                     - diff_flux) / diff_flux) * 100
        #plt.plot(spectral_index, residuals_log, marker='D',
        #             color='r', linewidth=0, ms=1)
    
   #indices = np.arange(len(energy_min))
   # for index in indices:
   #     plt.vlines(energy_min[index], -100, 800, color='k', linestyle=':')
    #plt.hlines(0, 0, 10, color='k')
    plt.ylabel('Residuals/%')
    plt.xlabel('Spectral Index')
    #plt.xlim([1, 10000])
    plt.show()
    
if __name__ == '__main__':
    make_x_plot()