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

class SedimentElement(Lagrangian3DArray):
    variables = Lagrangian3DArray.add_variables([
        ('settled', {'dtype': np.uint8,  # 0 is active, 1 is settled
                     'units': '1',
                     'default': 0}),
        ('terminal_velocity', {'dtype': np.float32,
                               'units': 'm/s',
                               'default': -0.001}),  # 1 mm/s negative buoyancy
        ('terminal_velocity_default', {'dtype': np.float32,
                               'units': 'm/s',
                               'default': -0.001}),  # 1 mm/s negative buoyancy
        ('grain_diameter', {'dtype': np.float32,
                 'units': 'm',
                 'default': 4e-6}),
        ('counter', {'dtype': np.uint8,
                      'units': '1',
                      'default': 0}),
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
        ('viscosity_molecular', {'dtype': np.float32,
                      'units': 'idk rn',
                      'default': 1.4e-3}),
        ('use_stokes', {'dtype': np.uint8,
                      'units': '1',
                      'default': 1}),
        ('times_resuspended', {'dtype': np.uint8,
                      'units': '1',
                      'default': 0})
        ])

    def move_elements(self, other, indices):
        super(Lagrangian3DArray, self).move_elements(other, indices)
        # Set terminal velocity to something calculated using Stokes Law
        grain_diameter = other.grain_diameter
        # This viscosity is in [kg/(ms)] so do not need to multiply by density seawater
        # viscosity = other.get_config('environment:molecular_viscosity')
        viscosity = other.viscosity_molecular
        gravity = -9.81
        rho_ocean = 1026.95
        rho_sed = other.rho_s
        print(f"input term vel: {other.terminal_velocity}")
        print(f"len input term vel: {len(other.terminal_velocity)}")
        term_vel = 2 / 9 * (rho_sed - rho_ocean) * gravity / viscosity * (grain_diameter / 2)**2 
        print(f"calculated: {term_vel}")
        not_stokes = 1- other.use_stokes
        term_vel = term_vel * other.use_stokes + other.terminal_velocity * not_stokes
        other.terminal_velocity = term_vel
        print(f"check terminal vel: {other.terminal_velocity}")
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

        self.vertical_mixing()  # Including buoyancy and settling

        # deactivated elements outside of bounds
        lon_min = -118.7264
        lon_max = -118.144003
        lat_min = 33.2672
        lat_max = 33.9328
        
        lons, lats = self.elements.lon, self.elements.lat
        out_of_bounds = (lons < lon_min) | (lons > lon_max) | (lats < lat_min) | (lats > lat_max)
        self.deactivate_elements(out_of_bounds, reason='out of bounds')

        upwards_moving_particles = self.elements.counter == 1
        self.elements.terminal_velocity[upwards_moving_particles] = self.elements.terminal_velocity_default[upwards_moving_particles]
        self.elements.counter[upwards_moving_particles] = 0

        self.resuspension()        

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
        # threshold = self.get_config('vertical_mixing:resuspension_threshold')
        # resuspending = np.logical_and(self.current_speed() > threshold, self.elements.moving==0)

        threshold = self.elements.tau_crit
        # Technically this calculation of bottom stress should only be done on particles near
        # the seafloor. So this calculation is not valid for the particles that have not settled.
        # I am leaving it this way though because it works better for finding the resuspending
        # index.
        
        bottom_stress = self.calc_bottom_stress()
        resuspending = np.logical_and(bottom_stress > threshold, self.elements.moving==0)
        
        if np.sum(resuspending) > 0:
            self.elements.times_resuspended[resuspending] = self.elements.times_resuspended[resuspending] + 1
            # Allow moving again
            self.elements.moving[resuspending] = 1
            self.elements.dz_bot[resuspending] = 99999.0

            self.elements.terminal_velocity[resuspending] = self.calc_upward_resuspension_velocity(bottom_stress, resuspending)
            self.elements.counter[resuspending] = 1

    def calc_bottom_stress(self):
        reader_names = list(self.env.readers.keys())
        reader_name = reader_names[0]
        u = self.env.readers[reader_name].get_variables('x_sea_water_velocity', time=self.time,
                      x=self.elements.lon, y=self.elements.lat, z=self.elements.z)
        v = self.env.readers[reader_name].get_variables('y_sea_water_velocity', time=self.time,
                      x=self.elements.lon, y=self.elements.lat, z=self.elements.z)

        u_vel = []
        v_vel = []

        for i in range(len(self.elements)):
            lon, lonidx = self.find_nearest(u['x'], self.elements.lon[i])
            lat, latidx = self.find_nearest(u['y'], self.elements.lat[i])
            z, zidx = self.find_nearest(u['z'], self.elements.z[i])
            u_vel.append(u['x_sea_water_velocity'][zidx, latidx, lonidx])
            v_vel.append(v['y_sea_water_velocity'][zidx, latidx, lonidx])

        u_vel = np.array(u_vel)
        v_vel = np.array(v_vel)            
        
        A_v = self.get_config('environment:fallback:ocean_vertical_diffusivity')
        dz_bot = self.elements.dz_bot
        r_b = 0
        c_d = 0.0021

        _2KE = u_vel**2 + v_vel**2

        bottom_stress_pseudo_energy_x = (2*A_v/dz_bot + r_b + c_d*np.sqrt(_2KE))*u_vel
        bottom_stress_pseudo_energy_y = (2*A_v/dz_bot + r_b + c_d*np.sqrt(_2KE))*v_vel
        bottom_stress_pseudo_energy_magnitude = np.sqrt(bottom_stress_pseudo_energy_x**2 + bottom_stress_pseudo_energy_y**2)
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
        E_0 = self.elements.E_0[resuspending]
        porosity = self.elements.porosity[resuspending]
        rho_s = self.elements.rho_s[resuspending]
        tau_crit = self.elements.tau_crit[resuspending]
        bot_stress = bottom_stress[resuspending]
        
        w = ((E_0 * (1-porosity))/rho_s) * ((bot_stress - tau_crit)/(bot_stress))
        return w

    def plot_property_nolegend(self, prop, filename=None, mean=False, labels=None, colors=None):
        """Basic function to plot time series of any element properties."""
        import matplotlib.pyplot as plt
        from matplotlib import dates

        hfmt = dates.DateFormatter('%d %b %Y %H:%M')
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
        days = [dt.date() for dt in times]
        times = days
        data = self.history[prop].T[0:len(times), :]
        if mean is True:  # Taking average over elements
            data = np.mean(data, axis=1)
        plt.plot(times, data)
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

    def plot_property_11(self, prop, filename=None, mean=False, labels=None, colors=None):
        """Basic function to plot time series of any element properties."""
        import matplotlib.pyplot as plt
        from matplotlib import dates
        from datetime import datetime

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
        days = [dt.date() for dt in times]
        times = days
        data = self.history[prop].T[0:len(times), :]
        
        if mean is True:  # Taking average over elements
            data = np.mean(data, axis=1)
            plt.plot(times, data)
        else:
            for col in range(len(data[0,:])):
                if colors is not None and labels is not None:
                    group = col // 11  # Determine the group (0, 1, 2, or 3)
                    color = colors[group]
                    if col % 11 == 0:  # Only label the first line in each group for the legend
                        plt.plot(days, data[:, col], color=color, label=labels[group])
                    else:
                        plt.plot(days, data[:, col], color=color)
                else:                
                    plt.plot(times, data[:,col], label=f"particle_{col}")
                    
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

    def plot_property_4(self, prop, filename=None, mean=False, labels=None, colors=None):
        """Basic function to plot time series of any element properties."""
        import matplotlib.pyplot as plt
        from matplotlib import dates
        from datetime import datetime
        import numpy as np  # Import numpy for np.mean
    
        super(OceanDrift, self).plot_property(prop, filename, mean)
    
        hfmt = dates.DateFormatter('%d %b %Y')
        fig = plt.figure()
        ax = fig.gca()
        ax.xaxis.set_major_formatter(hfmt)
        plt.xticks(rotation='vertical')
        
        start_time = self.start_time
        # In case start_time is unsupported cftime
        start_time = datetime(
            start_time.year, start_time.month,
            start_time.day, start_time.hour,
            start_time.minute, start_time.second
        )
        
        times = [
            start_time + n * self.time_step_output
            for n in range(self.steps_output)
        ]
        days = [dt.date() for dt in times]
        times = days
        data = self.history[prop].T[0:len(times), :]
        
        if mean:
            # Taking average over elements
            data = np.mean(data, axis=1)
            plt.plot(times, data)
        else:
            for col in range(len(data[0, :])):
                if colors is not None and labels is not None:
                    group = col
                    color = colors[group]
                    plt.plot(times, data[:, col], color=color, label=labels[group])
                else:
                    plt.plot(times, data[:, col], label=f"particle_{col}")
            
            plt.legend()
    
        plt.title(prop)
        plt.xlabel('Time  [UTC]')
        
        try:
            plt.ylabel('%s  [%s]' % (prop, self.elements.variables[prop]['units']))
        except:
            plt.ylabel(prop)
        
        plt.subplots_adjust(bottom=.3)
        plt.grid()
        
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename)
