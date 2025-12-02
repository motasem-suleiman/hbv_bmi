

import numpy as np
import yaml
import math


# function for the whole model
def run_routing(Qgen, 
                Qlist,
                Qgen_routed,
                MAXBAS):

    step = 0.005  # Integration step

    # Create sequence
    i = np.arange(0, MAXBAS + step, step)

    if i[-1] > MAXBAS:
        i = i[:-1]
    h = np.zeros(len(i))

    # Construct the triangular weighting function (first part)
    j = np.where(i < MAXBAS / 2)
    h[j] = step * (i[j] * 4 / MAXBAS ** 2)

    # Construct the triangular weighting function (second part)
    j = np.where(i >= MAXBAS / 2)
    h[j] = step * (4 / MAXBAS - i[j] * 4 / MAXBAS ** 2)

    # Adjust for extra weights for the last day if MAXBAS is non-integer
    stp = ((len(i)-1) / MAXBAS)
    I = np.arange(1, len(i), stp)
    I = np.append(I, len(i))

    MAXBAS_w = np.zeros(len(I))

    # Integration of function
    for k in range(1, len(I)):

        start_index = int(np.floor(I[k-1]))-1
        end_index = int(np.floor(I[k]))

        MAXBAS_w[k] = np.sum(h[start_index:end_index])  # Sum the slice of h

    # Normalize to ensure mass balance
    MAXBAS_w = MAXBAS_w[1:len(MAXBAS_w)] / np.nansum(MAXBAS_w[1:len(MAXBAS_w)])

    rev_MAXBAS_w = MAXBAS_w[::-1]

    q_alldays = rev_MAXBAS_w * Qgen
    
    Qlist = np.add(Qlist, q_alldays, out = Qlist)

    Qgen_routed[...] = Qlist[0]
    Qlist[...]= np.append(Qlist[1:], 0)


class HBV_ROUTING(object):
    
    def __init__(self, MAXBAS = 1, 
                       Qlist_init=0,
                       timestep=0, year=0, dayofyear=0, ):
        
        # --------------------------
        # Model Parameters
        # --------------------------
                
        # Routing Routine
        # ------------------------
        self._MAXBAS = MAXBAS

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
        self._Qgen_mm = np.zeros(1, dtype=float)
        
        # ------------------------------------------------------------
        # Output (create a space to hold their data)
        # ------------------------------------------------------------
        self._Qgen_routed_mm = np.zeros(1, dtype=float)

        
        # ------------------------------------------------------------
        # State Variables (water storage variables that gets updated)
        # ------------------------------------------------------------
        
        self._Qlist_mm = np.zeros(1, dtype=float)
        Qlist_tmp      = np.zeros(1, dtype=float)
        Qlist_tmp[0, ] = Qlist_init
        self._Qlist_mm = np.repeat(Qlist_tmp, int(np.ceil(MAXBAS)) )


    # Threshold Temperature (TT) Snow Routine
    @property
    def MAXBAS(self):
        """Rain-snow air temperature threshold when rs_method = 1."""
        return self._MAXBAS
    
    @MAXBAS.setter
    def MAXBAS(self, MAXBAS):
        """Set model rs_thresh."""
        self._MAXBAS = MAXBAS
    
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
    def Qgen_mm(self):
        """Current air temperature."""
        return self._Qgen_mm

    @Qgen_mm.setter
    def Qgen_mm(self, new_Qgen_mm):
        """Set air temperature."""
        self.Qgen_mm[:] = new_Qgen_mm
            
    # model output
    # ----------------------------
    @property
    def Qgen_routed_mm(self):
        """Current snowmelt."""
        return self._Qgen_routed_mm

    @Qgen_routed_mm.setter
    def Qgen_routed_mm(self, new_Qgen_routed_mm):
        """Set melt."""
        self._Qgen_routed_mm[:] = new_Qgen_routed_mm
        
    # State variables
    # ----------------------------
    
    # Threshold Temperature (TT) Snow Routine
    @property
    def Qlist_mm(self):
        """Current snow water equivalent."""
        return self._Qlist_mm

    @Qlist_mm.setter
    def Qlist_mm(self, Qlist_mm):
        """Set swe."""
        self._Qlist_mm[:] = Qlist_mm

    
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
        
        run_routing(
            self._Qgen_mm,
            self._Qlist_mm,
            self._Qgen_routed_mm,
            self._MAXBAS,
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