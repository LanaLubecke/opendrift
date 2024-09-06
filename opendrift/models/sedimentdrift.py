  # This file is part of OpenDrift.
#
# OpenDrift is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 2
#
# OpenDrift is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with OpenDrift.  If not, see <https://www.gnu.org/licenses/>.
#
# Copyright 2020, Knut-Frode Dagestad, MET Norway

"""
SedimentDrift is an OpenDrift module for drift and settling of sediments.
Based on work by Simon Weppe, MetOcean Solutions Ltd.
"""

import numpy as np
import logging; logger = logging.getLogger(__name__)
from opendrift.models.oceandrift import OceanDrift
from opendrift.models.oceandrift import Lagrangian3DArray
from opendrift.config import CONFIG_LEVEL_ESSENTIAL, CONFIG_LEVEL_BASIC, CONFIG_LEVEL_ADVANCED
from datetime import datetime
import math

class SedimentElement(Lagrangian3DArray):
    variables = Lagrangian3DArray.add_variables([
        ('settled', {'dtype': np.uint8,  # 0 is active, 1 is settled
                     'units': '1',
                     'default': 0}),
        ('terminal_velocity', {'dtype': np.float32,
                               'units': 'm/s',
                               'default': -0.001}),  # 1 mm/s negative buoyancy
        # use the terminal_velocity_default to restore downwards terminal velocity after
        # particle has been given an upwards one for resuspension
        ('terminal_velocity_default', {'dtype': np.float32,
                               'units': 'm/s',
                               'default': -0.001}),  
        ('grain_diameter', {'dtype': np.float32,
                 'units': 'm',
                 'default': 4e-6}),
        # used to give particle upward resuspension velocity for 1 timestep
        ('counter', {'dtype': np.uint8,
                      'units': '1',
                      'default': 0}),
        # Distance from particle to nearest cell center above, value set to 99999
        # if particle not settled
        ('dz_bot', {'dtype': np.float32,
                      'units': 'm',
                      'default': 99999.0}), 
        ('E_0', {'dtype': np.float32,
                      'units': 'kg*m^-3',
                      'default': 1e-5}),
        ('tau_crit', {'dtype': np.float32,
                      'units': 'Pa',
                      'default': 0.09}),
        ('rho_s', {'dtype': np.float32,
                      'units': 'kgm-3',
                      'default': 2000}),
        ('porosity', {'dtype': np.float32,
                      'units': 'unitless',
                      'default': 0.9}),
        # Molecular viscosity used for Stokes Law terminal velocity calculation.
        # Since this is a property of seawater, it would make sense for this
        # parameter to be stored in the model config. However, by putting it
        # in this location, the move_elements function has access to it, and
        # the move_elements function is where terminal velocity is calculated.
        ('viscosity_molecular', {'dtype': np.float32,
                      'units': 'kgm-1s-1',
                      'default': 1.4e-3}),
        # whether or not to use Stokes Law calculation or empirical value for terminal velocity
        ('use_stokes', {'dtype': np.uint8,
                      'units': '1',
                      'default': 1}),
        # keeps track of how many times a particle was resuspended
        ('times_resuspended', {'dtype': np.uint8,
                      'units': '1',
                      'default': 0})
        ])

    def move_elements(self, other, indices):
        super(Lagrangian3DArray, self).move_elements(other, indices)
        # Set terminal velocity to something calculated using Stokes Law
        grain_diameter = other.grain_diameter
        # This viscosity is in [kg/(ms)] so do not need to multiply by density seawater
        viscosity = other.viscosity_molecular
        gravity = -9.81
        # Since this method is a part of the Lagrangian3DArray class, it does not have access
        # to the sea_water_density function from physics methods, thus I just chose a default value.
        rho_ocean = 1026.95
        rho_sed = other.rho_s
        term_vel = 2 / 9 * (rho_sed - rho_ocean) * gravity / viscosity * (grain_diameter / 2)**2 
        not_stokes = 1 - other.use_stokes
        term_vel = term_vel * other.use_stokes + other.terminal_velocity * not_stokes
        other.terminal_velocity = term_vel
        other.terminal_velocity_default = term_vel
      


class SedimentDrift(OceanDrift):
    """Model for sediment drift, under development
    """

    ElementType = SedimentElement

    required_variables = {
        'x_sea_water_velocity': {'fallback': 0},
        'y_sea_water_velocity': {'fallback': 0},
        'sea_surface_height': {'fallback': 0},
        'upward_sea_water_velocity': {'fallback': 0},
        'x_wind': {'fallback': 0},
        'y_wind': {'fallback': 0},
        'sea_surface_wave_stokes_drift_x_velocity': {'fallback': 0},
        'sea_surface_wave_stokes_drift_y_velocity': {'fallback': 0},
        'sea_surface_wave_period_at_variance_spectral_density_maximum': {'fallback': 0},
        'sea_surface_wave_mean_period_from_variance_spectral_density_second_frequency_moment': {'fallback': 0},
        'land_binary_mask': {'fallback': None},
        'ocean_vertical_diffusivity': {'fallback': 0.02,
                                      'profiles': True},
        'ocean_mixed_layer_thickness': {'fallback': 50},
        'sea_floor_depth_below_sea_level': {'fallback': 10000},
        }

    def __init__(self, *args, **kwargs):
        """ Constructor of SedimentDrift module
        """

        super(SedimentDrift, self).__init__(*args, **kwargs)

        self._add_config({
            'vertical_mixing:resuspension_threshold': {
                'type': 'float',
                'default': 0.2,
                'min': 0,
                'max': 3,
                'units': 'm/s',
                'description':
                'Sedimented particles will be resuspended if bottom current shear exceeds this value.',
                'level': CONFIG_LEVEL_ESSENTIAL
            }})

        # Currently this config value is not being used as it is also saved in the
        # SedimentELement object. However, if the calculation for terminal velocity
        # were to be moved out of the SedimentElement class, it would be useful to
        # store molecular viscosity here.
        self._add_config({
            'environment:molecular_viscosity': {
                'type': 'float',
                'default': 1.4e-3,
                'min': 1e-3,
                'max': 2e-3,
                'units': 'kg(ms)-1',
                'description':
                'Sedimented particles will be resuspended if bottom current shear exceeds this value.',
                'level': CONFIG_LEVEL_ESSENTIAL
            }})


        # By default, sediments do not strand towards coastline
        # TODO: A more sophisticated stranding algorithm is needed
        self._set_config_default('general:coastline_action', 'previous')

        # Vertical mixing is enabled as default
        self._set_config_default('drift:vertical_mixing', True)
        # print(self.required_variables['x_sea_water_velocity'])
        # print(self.environment.x_sea_water_velocity)
    

    def update(self):
        """Update positions and properties of sediment particles.
        """
        # Advecting here all elements, but want to soon add
        # possibility of not moving settled elements, until
        # they are resuspended. May then need to send a boolean
        # array to advection methods below
        self.advect_ocean_current()

        self.vertical_advection()

        self.advect_wind()  # Wind shear in upper 10cm of ocean

        self.stokes_drift()

        if self.get_config('drift:vertical_mixing') is False:
            self.vertical_buoyancy()
        else:
            self.vertical_mixing() # including buoyancy and settling


        self.deactivate_elements_outofbounds()       

        upwards_moving_particles = self.elements.counter == 1
        # Restore downwards velocity to resuspended particles
        self.elements.terminal_velocity[upwards_moving_particles] = self.elements.terminal_velocity_default[upwards_moving_particles]
        self.elements.counter[upwards_moving_particles] = 0

        self.resuspension()

    def deactivate_elements_outofbounds(self):
        # This only works if the first reader you passed to the model
        # has the correct geographic bounds
        reader_names = list(self.env.readers.keys())
        reader_name = reader_names[0]
        lon_min = self.env.readers[reader_name].xmin
        lon_max = self.env.readers[reader_name].xmax
        lat_min = self.env.readers[reader_name].ymin
        lat_max = self.env.readers[reader_name].ymax
        
        lons, lats = self.elements.lon, self.elements.lat
        out_of_bounds = (lons < lon_min) | (lons > lon_max) | (lats < lat_min) | (lats > lat_max)
        self.deactivate_elements(out_of_bounds, reason='out of bounds')

    def bottom_interaction(self, seafloor_depth):
        """Sub method of vertical_mixing, determines settling"""
        # Elements at or below seafloor are settled, by setting
        # self.elements.moving to 0.
        # These elements will not move until eventual later resuspension.
        settling = np.logical_and(self.elements.z <= seafloor_depth, self.elements.moving==1)
        if np.sum(settling) > 0:
            logger.debug('Settling %s elements at seafloor' % np.sum(settling))
            self.elements.moving[settling] = 0
            self.get_distance_cell_center_above(settling)

    def resuspension(self):
        """Resuspending elements if current speed > .5 m/s"""
        ## Old resuspension condition
        # threshold = self.get_config('vertical_mixing:resuspension_threshold')
        # resuspending = np.logical_and(self.current_speed() > threshold, self.elements.moving==0)

        threshold = self.elements.tau_crit

        settled = self.elements.moving==0
        if np.sum(settled) > 0:
            print(f"num particles settled: {np.sum(settled)}")
            bottom_stress = self.calc_bottom_stress(settled)
            resuspending = np.logical_and(bottom_stress > threshold, settled)
        else:
            resuspending = settled
        
        if np.sum(resuspending) > 0:
            # Keep track of how many times particle has been resuspended
            self.elements.times_resuspended[resuspending] = self.elements.times_resuspended[resuspending] + 1
            # Allow moving again
            self.elements.moving[resuspending] = 1
            # Since not at bottom anymore, set distance to nearest cell center above to 99999
            self.elements.dz_bot[resuspending] = 99999.0
            # Give particle upwards velocity
            self.elements.terminal_velocity[resuspending] = self.calc_upward_resuspension_velocity(bottom_stress, resuspending)
            # Keep track of number of time steps particle has been in resuspension for
            self.elements.counter[resuspending] = 1

    def calc_bottom_stress(self, idxs):
        # gets horizontal velocity field at z position closest to particle
        # Equation for drag adapted from https://mitgcm.readthedocs.io/en/latest/algorithm/algorithm.html, equation 2.119
        reader_names = list(self.env.readers.keys())
        reader_name = reader_names[0]
        u = self.env.readers[reader_name].get_variables('x_sea_water_velocity', time=self.time,
                      x=self.elements.lon[idxs], y=self.elements.lat[idxs], z=self.elements.z[idxs])
        v = self.env.readers[reader_name].get_variables('y_sea_water_velocity', time=self.time,
                      x=self.elements.lon[idxs], y=self.elements.lat[idxs], z=self.elements.z[idxs])

        u_vel = []
        v_vel = []
        
        for i, idx in enumerate(idxs):
            if idx:
                lon, lonidx = self.find_nearest(u['x'], self.elements.lon[i])
                lat, latidx = self.find_nearest(u['y'], self.elements.lat[i])
                z, zidx = self.find_nearest(u['z'], self.elements.z[i])
                u_vel.append(u['x_sea_water_velocity'][zidx, latidx, lonidx])
                v_vel.append(v['y_sea_water_velocity'][zidx, latidx, lonidx])
            else:
                u_vel.append(0.0)
                v_vel.append(0.0)

        u_vel = np.array(u_vel)
        v_vel = np.array(v_vel)            
        
        A_v = self.get_config('environment:fallback:ocean_vertical_diffusivity')
        dz_bot = self.elements.dz_bot
        r_b = 0
        c_d = 0.0021

        _2KE = u_vel**2 + v_vel**2

        bottom_stress_pseudo_energy_magnitude = (2*A_v/dz_bot + r_b + c_d*np.sqrt(_2KE))*np.sqrt(u_vel**2 + v_vel**2)
        bottom_stress = self.sea_water_density()*bottom_stress_pseudo_energy_magnitude
        
        return bottom_stress     

    def find_nearest(self, array, value):
        idx = (np.abs(array - value)).argmin()
        return array[idx], idx

    def get_distance_cell_center_above(self, particles_seafloor):
        # This function assumes there is only one reader
        reader_names = list(self.env.readers.keys())
        reader_name = reader_names[0]
        z_grid = self.env.readers[reader_name].z
        floor_idx = np.array(np.where(particles_seafloor == 1))
        floor_idx = floor_idx.flatten()
        for i in floor_idx:
            z_grid_curr = z_grid[z_grid > self.elements.z[i]]
            closest_cell_center_height = z_grid_curr.min()
            self.elements.dz_bot[i] = closest_cell_center_height - self.elements.z[i]


    def calc_upward_resuspension_velocity(self, bottom_stress, resuspending):
        # Calculate the upwards velocity to give particle getting resuspended.
        # Equation adapted from https://doi.org/10.1061/JYCEAJ.0004937
        E_0 = self.elements.E_0[resuspending]
        porosity = self.elements.porosity[resuspending]
        rho_s = self.elements.rho_s[resuspending]
        tau_crit = self.elements.tau_crit[resuspending]
        bot_stress = bottom_stress[resuspending]
        
        w = ((E_0 * (1-porosity))/rho_s) * ((bot_stress - tau_crit)/(bot_stress))
        return w

    def plot_property_sedimentdrift(self, prop, filename=None, mean=False, labels=None, days=False, num_per_group=None, legend=True):
        """Basic function to plot time series of any element properties."""
        # This function adds onto the plot_property function that the basemodel already provides.
        # If you set days to true, the x axis labels wont include hours and seconds
        # Legend defaults to true. If you have more than one particle per label group.
        # you need to specify num_per_group and labels.
        import matplotlib.pyplot as plt
        from matplotlib import dates

        if not days:
            hfmt = dates.DateFormatter('%d %b %Y %H:%M')
        else:
            hfmt = dates.DateFormatter('%d %b %Y')
        fig = plt.figure()
        ax = fig.gca()
        ax.xaxis.set_major_formatter(hfmt)
        plt.xticks(rotation='vertical')
        start_time = self.start_time
        # In case start_time is unsupported cftime
        start_time = datetime(start_time.year, start_time.month,
                              start_time.day, start_time.hour,
                              start_time.minute, start_time.second)
        times = [
            start_time + n * self.time_step_output
            for n in range(self.steps_output)
        ]
        data = self.history[prop].T[0:len(times), :]
        if mean is True:  # Taking average over elements
            data = np.mean(data, axis=1)
            plt.plot(times, data)
        else:
            if num_per_group is not None:
                colors = plt.cm.viridis(np.linspace(0, 1, int(math.ceil(len(data[0,:]))/num_per_group)))
            for col in range(len(data[0,:])):
                if labels is not None:
                    group = col // num_per_group  # Determine the group (0, 1, 2, or 3)
                    color = colors[group]
                    if col % num_per_group == 0:  # Only label the first line in each group for the legend
                        plt.plot(times, data[:, col], color=color, label=labels[group])
                    else:
                        plt.plot(times, data[:, col], color=color)
                else:                
                    plt.plot(times, data[:,col], label=f"particle_{col}")

            if legend:
                plt.legend()
        plt.title(prop)
        plt.xlabel('Time  [UTC]')
        try:
            plt.ylabel('%s  [%s]' %
                       (prop, self.elements.variables[prop]['units']))
        except:
            plt.ylabel(prop)
        plt.subplots_adjust(bottom=.3)
        plt.grid()
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename)

