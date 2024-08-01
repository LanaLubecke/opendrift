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
                      'default': 0.9})
        ])


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

        upwards_moving_particles = self.elements.counter == 1
        self.elements.terminal_velocity[upwards_moving_particles] = -0.001
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
        if np.sum(self.elements.moving==0) > 0:
            print(f"max bottom stress of particles on bottom: {np.max(bottom_stress[self.elements.moving==0])}")
        
        if np.sum(resuspending) > 0:
            print("There are particles to resuspend")
            # Allow moving again
            self.elements.moving[resuspending] = 1
            self.elements.dz_bot[resuspending] = 99999.0
            # Suspend 1 cm above seafloor
            # self.elements.terminal_velocity[resuspending] = 0.01
            self.elements.terminal_velocity[resuspending] = self.calc_upward_resuspension_velocity(bottom_stress[resuspending])
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

        bottom_stress_pseudo_energy = (2*A_v/dz_bot + r_b + c_d*np.sqrt(_2KE))*u_vel
        bottom_stress = self.sea_water_density()*bottom_stress_pseudo_energy
        
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


    def calc_upward_resuspension_velocity(self, bottom_stress):
        E_0 = self.elements.E_0
        porosity = self.elements.porosity
        rho_s = self.elements.rho_s
        tau_crit = self.elements.tau_crit
        
        w = ((E_0 * (1-porosity))/rho_s) * ((bottom_stress - tau_crit)/(bottom_stress))
        return w



        
        
            
