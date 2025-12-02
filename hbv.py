
""" Hydrologiska Byråns Vattenbalansavdelning (HBV) model
    By: Motasem Suleiman Abualqumboz (Utah State University)
    Logan, UT 84341, USA
"""

"""
    The origional R code used to produce this Python code 
    for the HBV model were obtained from:
    https://github.com/VT-Hydroinformatics/16-Intro-Modeling-HBV/tree/master
"""


import numpy as np
import yaml
import math
import datetime


# function for the whole model
def run_hbv(precip, temp, PET,
            SP, WC, SM, SUZ, SLZ, Qlist,
            Qgen, AET, Qgen_routed,
            TT, SFCF, CFMAX, CFR, CWH, FC, beta, LP, k0, k1, k2, UZL, PERC, MAXBAS):
    
    """Run the HBV model for one time step to update the states and fluxes.

    1. Forcing input data
    ----------------------
    precip : ndarray
        Precipitation (input from forcing data).
    temp : ndarray
        Air temperature (input from forcing data).
    PET : ndarray
        Potential Evapotranspiration (input from forcing data).
        
    2. State variables (Gets updated each time this function is called)
    ----------------------
    SP : ndarray
        Snowpack
    WC : ndarray
        Liquid water (rainfall + snowmelt)
    SM : ndarray
        Soil Moisture
    SUZ : ndarray
        Storage in upper Groundwater zone
    SLZ : ndarray
        Storage in lower Groundwater zone
    
    
    3. Parameters
    ----------------------
    TT: float
        Threshold temperature
    SFCF: float
        Snowfall correction factor
    CFMAX: float
        Degree-day factor
    CFR: float
        Refreezing coefficient
    CWH: float
        Water holding capacity
    FC: float
        Maximum of SM (storage in soil box)
    beta: float
        Shape coefficient
    LP: float
        Threshold for reduction of evaporation (SM/FC)
    k0:float
        Recession coefficient (upper box in case of SUZ > UZL)
    k1: float
        Recession coefficient (upper box in case of SUZ < UZL)
    k2: float
        Recession coefficient (lower box)
    UZL: float
        Threshold parameter
    PERC: float
        Maximal flow from upper to lower box
    MAXBAS: float
        Routing, length of weighting function

    4. Outputs (Returns)
    ----------------------
    Qgen : ndarray
        Unrouted streamflow
    Qgen_routed : ndarray
        Routed streamflow
    AET : ndarray
        Actual Evapotranspiration
    """
    
    
    # 1. Snow module
    #---------------------------
    inc = 0
    if SP > 0:
        if precip > 0:
            if temp > TT:
                np.add(WC, precip, out=WC)
            else:
                np.add(SP, precip * SFCF, out=SP)
        if temp > TT:
            melt = CFMAX * (temp - TT)
            if melt > SP:
                inc = SP + WC
                WC.fill(0)
                SP.fill(0)
            else:
                np.subtract(SP, melt, out=SP)
                np.add(WC, melt, out=WC)
                if WC >= CWH * SP:
                    inc = WC - CWH * SP
                    WC[...] = CWH * SP
        else:
            refreeze = CFR * CFMAX * (TT - temp)
            if refreeze > WC:
                refreeze = WC
            np.add(SP, refreeze, out=SP)
            np.subtract(WC, refreeze, out=WC)
    else:
        if temp > TT:
            inc = precip
        else:
            np.add(SP, precip * SFCF, out=SP)

    # SWE = SP + WC

    # 2. Soil moisture module
    #---------------------------
    
    rQ = 0
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
                rQ += dQdP
                
        dQdP = (SM / FC) ** beta
        
        if dQdP > 1:
            dQdP = 1
        
        np.add(SM, (1 - dQdP) * y, out=SM)
        rQ += dQdP * y

    mean_SM = (SM + old_SM) / 2
    
    if (mean_SM < (LP*FC)):
        AET[...] = PET * mean_SM / (LP * FC)
    else:
        AET[...] = PET.copy()
    
    if (SP + WC) > 0:
        AET.fill(0)
        
    np.subtract(SM, AET, out=SM)
    
    if SM < 0:
        # soil = 0
        SM[...] = 0

    # soil = SM
    R = rQ
    w = inc
            
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
    # Storage = SUZ + SLZ + soil
    Storage = SUZ + SLZ + SM

    # 4. Routing Routine
    #---------------------------
    
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


# HBV Model Class

class HBV(object):
    
    def __init__(self, 
                 TT=0, SFCF=0, CFMAX=0, CFR=0, CWH=0,
                 FC=0, beta=0, LP=0, 
                 k0=0, k1=0, k2=0, UZL=0, PERC=0, MAXBAS = 1,
                 SP_init=0, WC_init=0, SM_init=0, SUZ_init=0, SLZ_init=0, Qlist_init=0 ,
                 timestep=0, year=0, dayofyear=0):
        
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
        
        # 2.Soil Routine
        # ------------------------
        self._FC    = FC
        self._beta  = beta
        self._LP    = LP
        
        # 3. Groundwater Routine
        # ------------------------
        self._k0    = k0
        self._k1    = k1
        self._k2    = k2
        self._UZL   = UZL
        self._PERC  = PERC
        
        # 4. Routing Routine
        # ------------------------
        self._MAXBAS= MAXBAS
        
        # --------------------------
        # Time information
        # --------------------------
        self._time      = 0.0
        self._elapsed_seconds = 0.0
        self._time_step = np.full(1, timestep)
        self._dayofyear = dayofyear
        self._year      = year
        
        # Calendar date (human-readable)
        self._current_date = None
        # self.current_date = datetime.datetime(self._year, 1, 1) + datetime.timedelta(days=self._dayofyear - 1)
        
        # --------------------------
        # Input forcing data
        # --------------------------
        self._ppt_mm = np.zeros(1, dtype=float) 
        self._tair_c = np.zeros(1, dtype=float)
        self._pet_mm = np.zeros(1, dtype=float)

        # ------------------------------------------------------------
        # State Variables (water storage variables that gets updated)
        # ------------------------------------------------------------
        self._SP_mm  = np.zeros(1, dtype=float)
        self._WC_mm  = np.zeros(1, dtype=float)
        self._SM_mm  = np.zeros(1, dtype=float)
        self._SUZ_mm = np.zeros(1, dtype=float)
        self._SLZ_mm = np.zeros(1, dtype=float)

        SP_tmp  = np.zeros(1, dtype=float)
        WC_tmp  = np.zeros(1, dtype=float)
        SM_tmp  = np.zeros(1, dtype=float)
        SUZ_tmp = np.zeros(1, dtype=float)
        SLZ_tmp = np.zeros(1, dtype=float)

        SP_tmp[0, ] = SP_init
        WC_tmp[0, ] = WC_init
        SM_tmp[0, ] = SM_init
        SUZ_tmp[0, ] = SUZ_init
        SLZ_tmp[0, ] = SLZ_init

        self._SP_mm  = SP_tmp
        self._WC_mm  = WC_tmp
        self._SM_mm  = SM_tmp
        self._SUZ_mm = SUZ_tmp
        self._SLZ_mm = SLZ_tmp
        
        self._Qlist_mm = np.zeros(1, dtype=float)
        Qlist_tmp      = np.zeros(1, dtype=float)
        Qlist_tmp[0, ] = Qlist_init
        self._Qlist_mm = np.repeat(Qlist_tmp, int(np.ceil(MAXBAS)) )
        
        # ------------------------------------------------------------
        # Output (create a space to hold their data)
        # ------------------------------------------------------------
        self._Qgen_mm = np.zeros(1, dtype=float)
        self._AET_mm = np.zeros(1, dtype=float)
        self._Qgen_routed_mm = np.zeros(1, dtype=float)

    
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
                        
    @property
    def MAXBAS(self):
        """Current precipitation."""
        return self._MAXBAS

    @MAXBAS.setter
    def MAXBAS(self, new_MAXBAS):
        """Set precipitation."""
        self._MAXBAS = new_MAXBAS
        

    # Time variables
    # ----------------------------
    
    # Current Model Time
    @property
    def time(self):
        """Current model time."""
        return self._time
    
    # Current Model elapsed_seconds
    @property
    def current_date(self):
        """Current model time."""
        # import datetime
        # return datetime.datetime(self._year, 1, 1) + datetime.timedelta(days=self._dayofyear - 1)
        return self._current_date

    
    # Model Time Step
    @property
    def time_step(self):
        """Model time step."""
        return self._time_step

    @time_step.setter
    def time_step(self, time_step):
        """Set time_step."""
        self._time_step[:] = time_step
    
    # Day of year (1 for January 1 and 365 for December 31)
    @property
    def dayofyear(self):
        """Current model day of year."""
        return self._dayofyear
    
    @dayofyear.setter
    def dayofyear(self, dayofyear):
        """Set model day of year."""
        self._dayofyear = dayofyear

    # Year
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
    
    # Atmospheric temperature
    @property
    def tair_c(self):
        """Current air temperature."""
        return self._tair_c

    @tair_c.setter
    def tair_c(self, new_tair_c):
        """Set air temperature."""
        self._tair_c[:] = new_tair_c
    
    # Precipitation
    @property
    def ppt_mm(self):
        """Current precipitation."""
        return self._ppt_mm

    @ppt_mm.setter
    def ppt_mm(self, new_ppt_mm):
        """Set precipitation."""
        self._ppt_mm[:] = new_ppt_mm
        
    # Precipitation
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
    
    # Unrouted generated runoff (mm day)
    @property
    def Qgen_mm(self):
        """Current snowmelt."""
        return self._Qgen_mm

    @Qgen_mm.setter
    def Qgen_mm(self, new_Qgen_mm):
        """Set melt."""
        self._Qgen_mm[:] = new_Qgen_mm
    
    # Actual Evapotranspiration (mm day)
    @property
    def AET_mm(self):
        """Current snowmelt."""
        return self._AET_mm

    @AET_mm.setter
    def AET_mm(self, new_AET_mm):
        """Set melt."""
        self._AET_mm[:] = new_AET_mm
        
    # Routed generated runoff (mm day)
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
    @property
    def SP_mm(self):
        """Current Snow Pack mm day"""
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

    @property
    def SM_mm(self):
        """Current snow water equivalent."""
        return self._SM_mm

    @SM_mm.setter
    def SM_mm(self, SM_mm):
        """Set SM."""
        self._SM_mm[:] = SM_mm
    
    @property
    def SUZ_mm(self):
        """Current snow water equivalent."""
        return self._SUZ_mm

    @SUZ_mm.setter
    def SUZ_mm(self, SUZ_mm):
        """Set SUZ."""
        self._SUZ_mm[:] = SUZ_mm
    
    @property
    def SLZ_mm(self):
        """Current snow water equivalent."""
        return self._SLZ_mm

    @SLZ_mm.setter
    def SLZ_mm(self, SLZ_mm):
        """Set SLZ."""
        self._SLZ_mm[:] = SLZ_mm
    
    @property
    def Qlist_mm(self):
        """Current snow water equivalent."""
        return self._Qlist_mm

    @Qlist_mm.setter
    def Qlist_mm(self, Qlist_mm):
        """Set Qlist."""
        self._Qlist_mm[:] = Qlist_mm
       
    #-------------------------------------------
    @classmethod
    def from_file_like(cls, file_like):
        """Create a HBV object from a file-like object.

        Parameters
        ----------
        file_like : file_like
            Input parameter file.

        Returns
        -------
        HBV
            A new instance of a HBV object.
        """
        config = yaml.safe_load(file_like)
        return cls(**config)
    
    
    def advance_in_time(self):
        
        """Run the HBV model for one time step (e.g. one day) """
        
        run_hbv(
            self._ppt_mm,
            self._tair_c,
            self._pet_mm,
            self._SP_mm,
            self._WC_mm,
            self._SM_mm,
            self._SUZ_mm,
            self._SLZ_mm,
            self._Qlist_mm,
            self._Qgen_mm,
            self._AET_mm,
            self._Qgen_routed_mm,
            self._TT,
            self._SFCF,
            self._CFMAX,
            self._CFR,
            self._CWH,
            self._FC,
            self._beta,
            self._LP,
            self._k0,
            self._k1,
            self._k2,
            self._UZL,
            self._PERC,
            self._MAXBAS,
        )
        
        # print(self._Qlist_mm)
        # print(self._Qgen_routed_mm)
                
        # Advance model clock forward in time
        self._time += self._time_step
        self._current_date = Dailydate = datetime.datetime(self._year, 1, 1) + datetime.timedelta(days=self._dayofyear - 1)

        if self._dayofyear == 365:    # check if day of year == 365
            if self._year % 4 != 0:   # if not leap year then increment to next year
                self._dayofyear = 1
                self._year += 1
            else:                     # otherwise (ie, it's a leap year) add 1 day
                self._dayofyear += 1  # increment time (self._time_step/86400)
        elif self._dayofyear == 366:  # check if end of a leap year and increment to next year
            self._dayofyear = 1
            self._year += 1
        else:                         # otherwise add one day to clock
            self._dayofyear += 1  # increment time (self._time_step/86400)