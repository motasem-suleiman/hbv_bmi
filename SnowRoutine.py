

import numpy as np
import yaml
import math


# function for the whole model
def run_snow(precip, temp,
            SP, WC,
            inc,
            TT, SFCF, CFMAX, CFR, CWH):
    
    # 1. Snow module
    #---------------------------
    inc.fill(0.0)
    if SP > 0:
        if precip > 0:
            if temp > TT:
                np.add(WC, precip, out=WC)
            else:
                np.add(SP, precip * SFCF, out=SP)
        if temp > TT:
            melt = CFMAX * (temp - TT)
            if melt > SP:
                inc[:] = SP + WC
                WC.fill(0)
                SP.fill(0)
            else:
                np.subtract(SP, melt, out=SP)
                np.add(WC, melt, out=WC)
                if WC >= CWH * SP:
                    inc[:] = WC - CWH * SP
                    WC[...] = CWH * SP
        else:
            refreeze = CFR * CFMAX * (TT - temp)
            if refreeze > WC:
                refreeze = WC
            np.add(SP, refreeze, out=SP)
            np.subtract(WC, refreeze, out=WC)
    else:
        if temp > TT:
            inc[:] = precip
        else:
            np.add(SP, precip * SFCF, out=SP)

    # print(inc)

    
class HBV_SNOW(object):
    
    def __init__(self, TT      = 0.0, SFCF      = 0.0, 
                       CFMAX   = 0.0, CFR       = 0.0,
                       CWH     = 0.0, SP_init   = 0.0,
                       WC_init = 0.0, timestep  = 0.0,
                       year    = 0.0, dayofyear = 0.0, ):
        
        # --------------------------
        # Model Parameters
        # --------------------------

        # 1.Snow Routine
        # ------------------------
        self._TT    = TT
        self._SFCF  = SFCF
        self._CFMAX = CFMAX
        self._CFR   = CFR
        self._CWH   = CWH
                        
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
        self._ppt_mm = np.zeros(1, dtype=float) 
        self._tair_c = np.zeros(1, dtype=float)
        
        # ------------------------------------------------------------
        # State Variables (water storage variables that gets updated)
        # ------------------------------------------------------------
        self._SP_mm  = np.zeros(1, dtype=float)
        self._WC_mm  = np.zeros(1, dtype=float)
        
        SP_tmp  = np.zeros(1, dtype=float)
        WC_tmp  = np.zeros(1, dtype=float)

        SP_tmp[0, ] = SP_init
        WC_tmp[0, ] = WC_init

        self._SP_mm  = SP_tmp
        self._WC_mm  = WC_tmp
        
        # ------------------------------------------------------------
        # Output (create a space to hold their data)
        # ------------------------------------------------------------
        self._inc_mm = np.zeros(1, dtype=float)

    # Threshold Temperature (TT) Snow Routine
    @property
    def TT(self):
        """Rain-snow air temperature threshold when rs_method = 1."""
        return self._TT
    
    @TT.setter
    def TT(self, TT):
        """Set model rs_thresh."""
        self._TT = TT
    
    # Snowfall correction factor (SFCF) Snow Routine
    @property
    def SFCF(self):
        """Rain-snow air temperature threshold when rs_method = 1."""
        return self._SFCF
    
    @SFCF.setter
    def SFCF(self, SFCF):
        """Set model rs_thresh."""
        self._SFCF = SFCF

    # Snowmelt degree-day factor (CFMAX) Snow Routine
    @property
    def CFMAX(self):
        """Rain-snow air temperature threshold when rs_method = 1."""
        return self._CFMAX
    
    @CFMAX.setter
    def CFMAX(self, CFMAX):
        """Set model rs_thresh."""
        self._CFMAX = CFMAX
    
    # Re-freezing coefficient (CFR) Snow Routine
    @property
    def CFR(self):
        """Rain-snow air temperature threshold when rs_method = 1."""
        return self._CFR
    
    @CFR.setter
    def CFR(self, CFR):
        """Set model rs_thresh."""
        self._CFR = CFR
    
    # Water holding capacity coefficient (CWH) Snow Routine
    @property
    def CWH(self):
        """Rain-snow air temperature threshold when rs_method = 1."""
        return self._CWH
    
    @CWH.setter
    def CWH(self, CWH):
        """Set model rs_thresh."""
        self._CWH = CWH
    
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
    def tair_c(self):
        """Current air temperature."""
        return self._tair_c

    @tair_c.setter
    def tair_c(self, new_tair_c):
        """Set air temperature."""
        self._tair_c[:] = new_tair_c

    @property
    def ppt_mm(self):
        """Current precipitation."""
        return self._ppt_mm

    @ppt_mm.setter
    def ppt_mm(self, new_ppt_mm):
        """Set precipitation."""
        self._ppt_mm[:] = new_ppt_mm
            
    # model output
    # ----------------------------
    @property
    def inc_mm(self):
        """Current snowmelt."""
        return self._inc_mm

    @inc_mm.setter
    def inc_mm(self, new_inc_mm):
        """Set melt."""
        self._inc_mm[:] = new_inc_mm
    
    # State variables
    # ----------------------------
    @property
    def SP_mm(self):
        """Current snow water equivalent."""
        return self._SP_mm

    @SP_mm.setter
    def SP_mm(self, SP_mm):
        """Set SP."""
        self._SP_mm[:] = SP_mm

    @property
    def WC_mm(self):
        """Current snow water equivalent."""
        return self._WC_mm

    @WC_mm.setter
    def WC_mm(self, WC_mm):
        """Set WC."""
        self._WC_mm[:] = WC_mm
    
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
        
        run_snow(
            self._ppt_mm,
            self._tair_c,
            self._SP_mm,
            self._WC_mm,
            self._inc_mm,
            self._TT,
            self._SFCF,
            self._CFMAX,
            self._CFR,
            self._CWH,
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