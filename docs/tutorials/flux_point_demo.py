"""Spectral plotting with gammapy.spectrum.flux_point
"""
import numpy as np
from astropy.table import Table
from gammapy.spectrum.flux_point import (compute_differential_flux_points,
                                         _energy_lafferty_power_law,
                                         _x_lafferty, _integrate)
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
    table2 = table1.copy()
    log_flux = compute_differential_flux_points(table2, 'log_center', y_method,
                                                function, spectral_index)
    return lafferty_flux, log_flux


def plot_flux_points(table, x, y, function, energy_min, energy_max, y_method):
    f, axarr = plt.subplots(2, sharex=True)
    lafferty_flux, log_flux = get_flux_tables(table, y_method, function,
                                              SPECTRAL_INDEX)

    axarr[0].loglog(x, (x ** 2) * y)
    axarr[0].loglog(lafferty_flux['ENERGY'],
                    ((lafferty_flux['ENERGY'] ** 2) * lafferty_flux['DIFF_FLUX']),
                    marker='D', color='k', linewidth=0, ms=5,
                    label='Lafferty Method')
    residuals_lafferty = (lafferty_flux['DIFF_FLUX']
                          - function(lafferty_flux['ENERGY'])) / function(lafferty_flux['ENERGY']) * 100
    axarr[0].loglog(log_flux['ENERGY'],
                    (log_flux['ENERGY'] ** 2) * log_flux['DIFF_FLUX'],
                    marker='D', color='r', linewidth=0, ms=5,
                    label='Log Center Method')
    axarr[0].legend(loc='lower left', fontsize=10)

    residuals_log = (log_flux['DIFF_FLUX'] - 
                     function(log_flux['ENERGY'])) / function(log_flux['ENERGY']) * 100
    axarr[1].semilogx(lafferty_flux['ENERGY'], residuals_lafferty, marker='D',
                      color='k', linewidth=0, ms=5)
    axarr[1].semilogx(log_flux['ENERGY'], residuals_log, marker='D',
                      color='r', linewidth=0, ms=5)
    indices = np.arange(len(energy_min))
    for index in indices:
        axarr[0].axvline(energy_min[index], 0, 1e6, color='k',
                         linestyle=':')
        axarr[1].axvline(energy_min[index], 0, 1e6, color='k',
                         linestyle=':')
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
    energies = np.array(_x_lafferty(energy_min, energy_max,
                                    spectral_model_function))
    diff_fluxes = spectral_model_function(energies)
    indices = np.array([0, 1, 2, 3, 4])
    int_flux = power_law_integral_flux(diff_fluxes, (indices+1),
                                       energies, energy_min, energy_max)
    special = lambda x: np.log(x)
    #import IPython; IPython.embed()
    int_flux[0] = np.abs(_integrate([energy_min[0]],
                                    [energy_max[0]], special)[0])
    # Put data into table
    table = Table()
    table['ENERGY_MIN'] = energy_min
    table['ENERGY_MAX'] = energy_max
    table['INT_FLUX'] = int_flux
    plt = plot_flux_points(table, x, y, spectral_model_function,
                           energy_min, energy_max, 'power_law')
    plt.legend()
    plt.show()

def compute_flux_residuals(gamma_true, gamma_reco, method):
    # These examples are picked as an energy bin
    energy_min = np.array([300])
    energy_max = np.array([1000])
    # Realistic example
    energies = np.array([371.2])
    diff_flux =  power_law_eval(energies, 1, gamma_true, 1)
    # `True' differential & integral flux
    int_flux = power_law_integral_flux(diff_flux, gamma_true,
                                       energies, energy_min, energy_max)
    lafferty_flux = np.zeros(int_flux.size)
    log_flux = np.zeros(int_flux.size)
    indices = np.arange(int_flux.size)
    for index in indices:
        new_int_flux = int_flux.reshape(indices.shape)
        table = Table()
        table['ENERGY_MIN'] = energy_min
        table['ENERGY_MAX'] = energy_max
        table['INT_FLUX'] = new_int_flux[index]
        gamma_reco = gamma_reco.reshape(indices.shape)
        lafferty, log = get_flux_tables(table, 'power_law', None,
                                        gamma_reco[index])
        lafferty_flux[index]=lafferty['DIFF_FLUX'][0]
        log_flux[index]=log['DIFF_FLUX'][0]
    diff_flux = diff_flux.reshape((diff_flux.size, ))
    lafferty_residuals = ((lafferty_flux-diff_flux)/diff_flux)*100
    log_residuals = ((log_flux-diff_flux)/diff_flux)*100
    if method == 'lafferty':
        return lafferty_residuals.reshape(int_flux.shape)
    elif method == 'log_center':
        return log_residuals.reshape(int_flux.shape)
    elif method == 'ratio':
        return (lafferty_residuals/log_residuals).reshape(int_flux.shape)
    
    
def residuals_image():
    gamma_true = np.arange(1.01, 10, 1)
    gamma_reco = np.arange(1.01, 10, 1)
    gamma_true, gamma_reco = np.meshgrid(gamma_true, gamma_reco)
    residuals_lafferty = compute_flux_residuals(gamma_true,gamma_reco,
                                                method='lafferty')
    residuals_logcenter = compute_flux_residuals(gamma_true, gamma_reco,
                                                 method='log_center')
    residuals_ratio = compute_flux_residuals(gamma_true, gamma_reco,
                                             method='ratio')
    f, axarr = plt.subplots(nrows=1, ncols=3)
    axarr.flat[0].imshow(residuals_lafferty, interpolation='nearest',
                         extent=[np.min(gamma_reco), np.max(gamma_reco),
                                 np.min(gamma_true), np.max(gamma_true)],
                         origin="lower")
    axarr.flat[0].set_xlabel('Assumed Spectral Index')
    axarr.flat[0]. set_ylabel('True Spectral Index')
    axarr.flat[1].imshow(residuals_logcenter, interpolation='nearest',
                         extent=[np.min(gamma_reco), np.max(gamma_reco),
                                 np.min(gamma_true), np.max(gamma_true)],
                         origin="lower")
    axarr.flat[1].set_xlabel('Assumed Spectral Index')
    axarr.flat[1]. set_ylabel('True Spectral Index')
    axarr.flat[2].imshow(residuals_ratio, interpolation='nearest',
                         extent=[np.min(gamma_reco), np.max(gamma_reco),
                                 np.min(gamma_true), np.max(gamma_true)],
                         origin="lower")
    axarr.flat[2].set_xlabel('Assumed Spectral Index')
    axarr.flat[2]. set_ylabel('True Spectral Index')
    plt.show()

    
def make_x_plot():
    energy_min = np.array([300])
    energy_max = np.array([1000])
    energies = np.array(_energy_lafferty_power_law(energy_min, energy_max,
                                                   SPECTRAL_INDEX))
    diff_flux =  power_law_eval(energies, 1, SPECTRAL_INDEX, 1)
    # `True' differential & integral fluxes
    int_flux = power_law_integral_flux(diff_flux, SPECTRAL_INDEX,
                                       energies, energy_min, energy_max)
    # Put data into table
    table = Table()
    table['ENERGY_MIN'] = energy_min
    table['ENERGY_MAX'] = energy_max
    table['INT_FLUX'] = int_flux
    lafferty_array = []  
    log_array = []
    spectral_indices = np.arange(1.1, 6, 0.01)
    for spectral_index in spectral_indices:
        lafferty_flux, log_flux = get_flux_tables(table, 'power_law', None,
                                                  spectral_index)
        residuals_lafferty = ((np.log(lafferty_flux['ENERGY'])
                              - np.log(energy_min)) / (np.log(energy_max)-np.log(energy_min)))
        residuals_log = ((np.log(log_flux['ENERGY'])
                              - np.log(energy_min)) / (np.log(energy_max)-np.log(energy_min)))
        lafferty_array.append(residuals_lafferty[0])
        log_array.append(residuals_log[0])
    plt.plot(spectral_indices, lafferty_array, color='k',
             linewidth=1, ms=0, label='Lafferty Method')
    plt.plot(spectral_indices, log_array, color='r',
             linewidth=1, ms=0, label='Log Center Method')
    plt.legend()
    plt.ylabel('X position in bin')
    plt.xlabel('Guessed spectral Index')
    plt.show()
    
if __name__ == '__main__':
    residuals_image()