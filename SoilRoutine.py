

import numpy as np
import yaml
import math


# function for Soil Routine in the HBV model
def run_soil(PET, inc, SP, WC,
             SM,
             AET, rQ, 
             FC, beta, LP):

    # Soil moisture module
    #---------------------------

    runoff = 0
    old_SM = SM.copy()
    
    if inc > 0:
        if inc < 1:
            y = inc
        else:
            m = int(inc)
            y = inc - m
            
            for j in range(1, m+1, 1):
                dQdP = (SM / FC) ** beta
                
                if dQdP > 1:
                    dQdP = 1
                                    
                np.add(SM, 1 - dQdP, out=SM)
                runoff += dQdP
                
        dQdP = (SM / FC) ** beta
        
        if dQdP > 1:
            dQdP = 1
        
        np.add(SM, (1 - dQdP) * y, out=SM)
        runoff += dQdP * y

    mean_SM = (SM + old_SM) / 2
    
    if (mean_SM < (LP*FC)):
        AET[...] = PET * mean_SM / (LP * FC)
    else:
        AET[...] = PET.copy()
    
    if (SP + WC) > 0:
        AET.fill(0)
        
    np.subtract(SM, AET, out=SM)
    
    if SM < 0:
        SM[...] = 0

    rQ[...] = runoff
    R = runoff
    w = inc


class HBV_SOIL(object):
    
    def __init__(self,
                 FC        = 0.0, beta    = 0.0, LP        = 0.0,
                 SM_init = 0.0,
                 timestep  = 0.0, year    = 0.0, dayofyear = 0.0, ):
        
        # --------------------------
        # Model Parameters
        # --------------------------
               
        # 2.Soil Routine
        # ------------------------
        self._FC    = FC
        self._beta  = beta
        self._LP    = LP
        
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
        self._inc_mm = np.zeros(1, dtype=float) 
        self._SP_mm = np.zeros(1, dtype=float)
        self._pet_mm = np.zeros(1, dtype=float)
        self._WC_mm = np.zeros(1, dtype=float)

        
        # ------------------------------------------------------------
        # State Variables (water storage variables that gets updated)
        # ------------------------------------------------------------
        
        self._SM_mm  = np.zeros(1, dtype=float)
        SM_tmp  = np.zeros(1, dtype=float)
        SM_tmp[0, ] = SM_init
        self._SM_mm  = SM_tmp

        # ------------------------------------------------------------
        # Output (create a space to hold their data)
        # ------------------------------------------------------------
        self._AET_mm = np.zeros(1, dtype=float)
        self._rQ_mm = np.zeros(1, dtype=float)

    
    # Maximum soil moisture storage (FC) Soil Routine
    @property
    def FC(self):
        """Rain-snow air temperature threshold when rs_method = 1."""
        return self._FC
    
    @FC.setter
    def FC(self, FC):
        """Set model rs_thresh."""
        self._FC = FC
        
    # Shape coefficient (beta) Soil Routine
    @property
    def beta(self):
        """Rain-snow air temperature threshold when rs_method = 1."""
        return self._beta
    
    @beta.setter
    def beta(self, beta):
        """Set model rs_thresh."""
        self._beta = beta
        
    # Threshold for reduction of evaporation (SM/FC)(beta) Soil Routine
    @property
    def LP(self):
        """Rain-snow air temperature threshold when rs_method = 1."""
        return self._LP
        
    @LP.setter
    def LP(self, LP):
        """Set model rs_thresh."""
        self._LP = LP

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
    def inc_mm(self):
        """Current precipitation."""
        return self._inc_mm

    @inc_mm.setter
    def inc_mm(self, new_inc_mm):
        """Set precipitation."""
        self._inc_mm[:] = new_inc_mm
     
    @property
    def SP_mm(self):
        """Current precipitation."""
        return self._SP_mm

    @SP_mm.setter
    def SP_mm(self, new_SP_mm):
        """Set precipitation."""
        self._SP_mm[:] = new_SP_mm
        
    @property
    def WC_mm(self):
        """Current precipitation."""
        return self._WC_mm

    @WC_mm.setter
    def WC_mm(self, new_WC_mm):
        """Set precipitation."""
        self._WC_mm[:] = new_WC_mm   

    
    @property
    def pet_mm(self):
        """Current precipitation."""
        return self._pet_mm

    @pet_mm.setter
    def pet_mm(self, new_pet_mm):
        """Set precipitation."""
        self._pet_mm[:] = new_pet_mm        


    # model output
    # ----------------------------    
    @property
    def rQ_mm(self):
        """Current snowmelt."""
        return self._rQ_mm

    @rQ_mm.setter
    def rQ_mm(self, new_rQ_mm):
        """Set melt."""
        self._rQ_mm[:] = new_rQ_mm
    
    @property
    def AET_mm(self):
        """Current snowmelt."""
        return self._AET_mm

    @AET_mm.setter
    def AET_mm(self, new_AET_mm):
        """Set melt."""
        self._AET_mm[:] = new_AET_mm
        
    # State variables
    # ----------------------------
   
    @property
    def SM_mm(self):
        """Current snow water equivalent."""
        return self._SM_mm

    @SM_mm.setter
    def SM_mm(self, SM_mm):
        """Set SM."""
        self._SM_mm[:] = SM_mm
    
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
        
        run_soil(
            self._pet_mm,
            self._inc_mm,
            self._SP_mm,
            self._WC_mm,
            self._SM_mm,
            self._AET_mm,
            self._rQ_mm,
            self._FC,
            self._beta,
            self._LP,
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