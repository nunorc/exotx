
""" Classes and aliases definitions. """

from typing import List
import logging

import lightkurve as lk
from lightkurve import periodogram

logger = logging.getLogger(__name__)

class Args:
    """ A class for storing arguments related with an object.
    This is mainly used for declaring information for the retrieve function.

    Attributes:
      target (str): target name (e.g. 'Kepler 210')
      mission (str): mission, defaults to 'Kepler'
      quarters ([int]): list of quarters
      cadence (str): 'short' or 'long' cadence
      pdc (bool): use PDF data

    """
    def __init__(self,
                 target: str = '',
                 mission: str = 'Kepler',
                 quarters: List[int] = [1],
                 cadence: str = 'long',
                 pdc: bool = True):
        """ Init class instance. """
        self.target = target
        self.mission = mission
        self.quarters = quarters
        self.cadence = cadence
        self.pdc = pdc

    def __str__(self):
        return self._to_string()

    def __repr__(self):
        return f"Args({ self._to_string() })"

    def _to_string(self):
        return f"target='{ self.target }', \
                 mission='{ self.mission }', \
                 quarters={ self.quarters }, \
                 cadence='{ self.cadence }', \
                 pdc={ self.pdc}"

class Params:
    """ A wrapper class for storing parameters related with some operations.

    Attributes:
      dict: list of pairs key, value

    """
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __str__(self):
        return self._to_string()

    def __repr__(self):
        return f"Params({ self._to_string() })"

    def _to_string(self):
        pairs = []
        for key in dir(self):
            if not key.startswith('_'):
                pairs.append(key + '=' + str(getattr(self, key)))
        return ", ".join(pairs)

    def _set(self, name, value):
        setattr(self, name, value)

class Fit:
    """ A wrapper class for storing a fit related with some operations.

    Attributes:
      dict: list of pairs key, value

    """
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __str__(self):
        return self._to_string()

    def __repr__(self):
        return f"Fit({ self._to_string() })"

    def _to_string(self):
        pairs = []
        for key in dir(self):
            if not key.startswith('_'):
                pairs.append(key + '=' + str(getattr(self, key)))
        return ", ".join(pairs)

    def _set(self, name, value):
        setattr(self, name, value)

# type aliases
LightCurve = lk.LightCurve
Transit = lk.LightCurve
Periodogram = periodogram.Periodogram
