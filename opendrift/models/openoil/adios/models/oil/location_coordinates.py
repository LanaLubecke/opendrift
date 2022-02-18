"""
Location Coordinates Object

When saved out as JSON, this is a subset of geoJSON

i.e. a geoJSON geometry, but can only be a Point or a Polygon.

So either::

      {
        'type': 'Point',
        'coordinates': [0, 0]
      }

or::

      {
        'type': 'Polygon',
        'coordinates': [[[-5e6, -1e6], [-4e6, 1e6], [-3e6, -1e6]]]
      }

"""
from dataclasses import dataclass, field

from ..common.utilities import dataclass_to_json

@dataclass_to_json
@dataclass
class LocationCoordinates:
    type: str
    coordinates: list

    def __post_init__(self):
        '''
        Put any validation code here (__init__() is auto-generated by the
        dataclass decorator, and it will clobber any attempt to overload
        the __init__.)
        '''
        coords = self.coordinates
        if self.type == "Point":
            try:
                if len(coords) != 2:
                    raise ValueError("Point type must have two coordinates")
            except TypeError:
                raise ValueError("Point type must have two coordinates")
            try:
                self.coordinates = (float(coords[0]), float(coords[1]))
            except (ValueError, TypeError):
                raise ValueError("Point type coordinates must be a pair of numbers")
        elif self.type == "Polygon":
            try:
                ring = coords[0]
                x, y = ring[0]
            except TypeError:
                raise ValueError("Polygon coordinates must be a list of lists of points")
            if len(ring) < 4:
                raise ValueError("Polygon coordinates must be a least four points")
            if ring[0] != ring[-1]:
                raise ValueError("First and last point of a polygon must be equal")
            for p in ring:
                if len(p) != 2:
                    raise ValueError("All points of a polygon must have two coordinates")
        else:
            raise ValueError('LocationCoordinates type must be either "Point" or "Polygon"')

