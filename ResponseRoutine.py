

import numpy as np
import yaml
import math


# function for Response (Groundwater) Routine in the HBV model
def run_response(rQ, SM, 
                 SUZ, SLZ,
                 Qgen,
                 k0, k1, k2, UZL, PERC):

    # 3. Groundwater storage module
    #---------------------------
    np.add(SUZ, rQ, out=SUZ)
    if (SUZ - PERC) < 0:
        np.add(SLZ, SUZ, out=SLZ)
        SUZ.fill(0)
    else:
        np.add(SLZ, PERC, out=SLZ)
        np.subtract(SUZ, PERC, out=SUZ)
    
    if (SUZ < UZL):
        Q_STZ = 0
    else:
        Q_STZ = (SUZ - UZL) * k0
        
    Q_SUZ = SUZ * k1
    Q_SLZ = SLZ * k2
    np.subtract(SUZ, Q_SUZ + Q_STZ, out=SUZ)
    np.subtract(SLZ, Q_SLZ, out=SLZ)
    
    Qgen[...] = Q_STZ + Q_SUZ + Q_SLZ
    Storage = SUZ + SLZ + SM


class HBV_RESPONSE(object):
    
    def __init__(self, k0        = 0.0, k1  = 0.0,
                       k2        = 0.0, UZL = 0.0, 
                       PERC      = 0.0,
                       SUZ_init  = 0.0, SLZ_init  = 0.0,
                       timestep  = 0.0, year = 0.0,
                       dayofyear = 0.0, ):
        
        # --------------------------
        # Model Parameters
        # --------------------------
        
        # 3. Groundwater Routine
        # ------------------------
        self._k0    = k0
        self._k1    = k1
        self._k2    = k2
        self._UZL   = UZL
        self._PERC  = PERC
                
        # --------------------------
        # Time information
        # --------------------------
        self._time = 0.0
        self._time_step    = np.full(1, timestep)
        self._dayofyear    = dayofyear
        self._year         = year
        
        # --------------------------
        # Input forcing data
        # --------------------------
        self._rQ_mm = np.zeros(1, dtype=float) 
        self._SM_mm = np.zeros(1, dtype=float) 

        # ------------------------------------------------------------
        # State Variables (water storage variables that gets updated)
        # ------------------------------------------------------------
        self._SUZ_mm = np.zeros(1, dtype=float)
        self._SLZ_mm = np.zeros(1, dtype=float)
        
        SUZ_tmp = np.zeros(1, dtype=float)
        SLZ_tmp = np.zeros(1, dtype=float)

        SUZ_tmp[0, ] = SUZ_init
        SLZ_tmp[0, ] = SLZ_init

        self._SUZ_mm = SUZ_tmp
        self._SLZ_mm = SLZ_tmp

        # ------------------------------------------------------------
        # Output (create a space to hold their data)
        # ------------------------------------------------------------
        self._Qgen_mm = np.zeros(1, dtype=float)
    
    @property
    def k0(self):
        """Rain-snow air temperature threshold when rs_method = 1."""
        return self._k0
        
    @k0.setter
    def k0(self, k0):
        """Set model rs_thresh."""
        self._k0 = k0
    
    @property
    def k1(self):
        """Rain-snow air temperature threshold when rs_method = 1."""
        return self._k1
        
    @k1.setter
    def k1(self, k1):
        """Set model rs_thresh."""
        self._k1 = k1
    
    @property
    def k2(self):
        """Rain-snow air temperature threshold when rs_method = 1."""
        return self._k2
        
    @k2.setter
    def k2(self, k2):
        """Set model rs_thresh."""
        self._k2 = k2
    
    @property
    def UZL(self):
        """Rain-snow air temperature threshold when rs_method = 1."""
        return self._UZL
        
    @UZL.setter
    def UZL(self, UZL):
        """Set model rs_thresh."""
        self._UZL = UZL
    
    @property
    def PERC(self):
        """Rain-snow air temperature threshold when rs_method = 1."""
        return self._PERC
        
    @PERC.setter
    def PERC(self, PERC):
        """Set model rs_thresh."""
        self._PERC = PERC
    
    # Time variables
    # ----------------------------

    @property
    def time(self):
        """Current model time."""
        return self._time
    
    @property
    def time_step(self):
        """Model time step."""
        return self._time_step

    @time_step.setter
    def time_step(self, time_step):
        """Set time_step."""
        self._time_step[:] = time_step

    @property
    def dayofyear(self):
        """Current model day of year."""
        return self._dayofyear

    @dayofyear.setter
    def dayofyear(self, dayofyear):
        """Set model day of year."""
        self._dayofyear = dayofyear

    @property
    def year(self):
        """Current model year."""
        return self._year

    @year.setter
    def year(self, year):
        """Set model year."""
        self._year = year
        
    # forcing data
    # ----------------------------
    @property
    def rQ_mm(self):
        """Current air temperature."""
        return self._rQ_mm

    @rQ_mm.setter
    def rQ_mm(self, new_rQ_mm):
        """Set air temperature."""
        self._rQ_mm[:] = new_rQ_mm
        
    @property
    def SM_mm(self):
        """Current air temperature."""
        return self._SM_mm

    @SM_mm.setter
    def SM_mm(self, new_SM_mm):
        """Set air temperature."""
        self._SM_mm[:] = new_SM_mm

            
    # model output
    # ----------------------------
    @property
    def Qgen_mm(self):
        """Current snowmelt."""
        return self._Qgen_mm

    @Qgen_mm.setter
    def Qgen_mm(self, new_Qgen_mm):
        """Set melt."""
        self._Qgen_mm[:] = new_Qgen_mm
        
    # State variables
    # ----------------------------
        
    @property
    def SUZ_mm(self):
        """Current snow water equivalent."""
        return self._SUZ_mm

    @SUZ_mm.setter
    def SUZ_mm(self, SUZ_mm):
        """Set swe."""
        self._SUZ_mm[:] = SUZ_mm
    
    @property
    def SLZ_mm(self):
        """Current snow water equivalent."""
        return self._SLZ_mm

    @SLZ_mm.setter
    def SLZ_mm(self, SLZ_mm):
        """Set swe."""
        self._SLZ_mm[:] = SLZ_mm

    
    @classmethod
    def from_file_like(cls, file_like):
        """Create a Snow object from a file-like object.

        Parameters
        ----------
        file_like : file_like
            Input parameter file.

        Returns
        -------
        Snow
            A new instance of a Snow object.
        """
        config = yaml.safe_load(file_like)
        return cls(**config)
    
    
    def advance_in_time(self):
        """Calculate new temperatures for the next time step."""
        
        run_response(
            self._rQ_mm,
            self._SM_mm,
            self._SUZ_mm,
            self._SLZ_mm,
            self._Qgen_mm,
            self._k0,
            self._k1,
            self._k2,
            self._UZL,
            self._PERC,
        )
                
        # Advance model clock forward in time
        self._time += self._time_step

        if self._dayofyear == 365:    # check if day of year == 365
            if self._year % 4 != 0:   # if not leap year then increment to next year
                self._dayofyear = 1
                self._year += 1
            else:                     # otherwise (ie, it's a leap year) add 1 day
                self._dayofyear += 1  # increment time
        elif self._dayofyear == 366:  # check if end of a leap year and increment to next year
            self._dayofyear = 1
            self._year += 1
        else:                         # otherwise add one day to clock
            self._dayofyear += 1  # increment time