import numpy as np
from HARKcore import AgentType, NullFunc
from HARKutilities import warnings  # Because of "patch" to warnings modules
from HARKutilities import calculateMeanOneLognormalDiscreteApprox, addDiscreteOutcomeConstantMean, createFlatStateSpaceFromIndepDiscreteProbs, setupGridsExpMult, CRRAutility, CRRAutilityP, CRRAutilityPP, CRRAutilityP_inv, CRRAutility_invP, CRRAutility_inv
from HARKinterpolation import Cubic1DInterpDecay, ConstrainedComposite, LinearInterp
from HARKsimulation import generateMeanOneLognormalDraws, generateBernoulliDraws
from scipy.optimize import newton, brentq
from copy import deepcopy

utility = CRRAutility
utilityP = CRRAutilityP
utilityPP = CRRAutilityPP
utilityP_inv = CRRAutilityP_inv
utility_invP = CRRAutility_invP
utility_inv = CRRAutility_inv

# =====================================================================
# === Classes and functions used to solve consumption-saving models ===
# =====================================================================

class ConsumerSolution():
    '''
    A class representing the solution of a single period of a consumption-saving
    problem.  The solution must include a consumption function, but may also include
    the minimum allowable money resources m_underbar, expected human wealth gothic_h,
    and the lower and upper bounds on the MPC kappa_min and kappa_max.  A value
    function can also be included, as well as marginal value and marg marg value.
    '''

    def __init__(self, cFunc=NullFunc, vFunc=NullFunc, vPfunc=NullFunc, vPPfunc=NullFunc, m_underbar=None, gothic_h=None, kappa_min=None, kappa_max=None):
        '''
        The constructor for a new ConsumerSolution object.
        '''
        self.cFunc = cFunc
        self.vFunc = vFunc
        self.vPfunc = vPfunc
        self.vPPfunc = vPPfunc
        self.m_underbar = m_underbar
        self.gothic_h = gothic_h
        self.kappa_min = kappa_min
        self.kappa_max = kappa_max

    def distance(self,solution_other):
        '''
        Returns the distance between single period solutions as the distance
        between their consumption functions.
        '''
        return self.cFunc.distance(solution_other.cFunc)




def consumptionSavingSolverEXOG(solution_tp1,income_distrib,p_zero_income,survival_prob,beta,rho,R,Gamma,constrained,a_grid,calc_vFunc,cubic_splines):
    '''
    Solves a single period of a standard consumption-saving problem, representing
    the consumption function as a cubic spline interpolation if cubic_splines is
    True and as a linear interpolation if it is False.  Problem is solved using
    the method of exogenous gridpoints (the "typical way" aka the "slow way").

    Parameters:
    -----------
    solution_tp1: ConsumerSolution
        The solution to the following period.
    income_distrib: [[float]]
        A list containing three lists of floats, representing a discrete approximation to the income process between
        the period being solved and the one immediately following (in solution_tp1).  Order: psi, xi, probs
    p_zero_income: float
        The probability of receiving zero income in the succeeding period.
    survival_prob: float
        Probability of surviving to succeeding period.
    beta: float
        Discount factor between this period and the succeeding period.
    rho: float
        The coefficient of relative risk aversion
    R: float
        Interest factor on assets between this period and the succeeding period: w_tp1 = a_t*R
    Gamma: float
        Expected growth factor for permanent income between this period and the succeeding period.
    constrained: bool
        Whether this individual may hold net negative assets at the end of the period.
    a_grid: [float]
        A list of beginning-of-period m_t values at which to solve for optimal consumption.
    calc_vFunc: Boolean
        An indicator for whether the value function should be computed and included
        in the reported solution
    cubic_splines: Boolean
        An indicator for whether the solver should use cubic or linear interpolation
    

    Returns:
    -----------
    solution_t: ConsumerSolution
        The solution to this period's problem, obtained using the method of endogenous gridpoints.
    '''

    # Define utility and value functions
    if calc_vFunc:
        u = lambda c : utility(c,gam=rho)
        uinv = lambda u : utility_inv(u,gam=rho)
        uinvP = lambda u : utility_invP(u,gam=rho)
        vFunc_tp1 = solution_tp1.vFunc
    uP = lambda c : utilityP(c,gam=rho)
    uPP = lambda c : utilityPP(c,gam=rho)
    #uPinv = lambda u : utilityP_inv(u,gam=rho)

    # Set and update values for this period
    effective_beta = beta*survival_prob
    psi_tp1 = income_distrib[1]
    xi_tp1 = income_distrib[2]
    prob_tp1 = income_distrib[0]
    vPfunc_tp1 = solution_tp1.vPfunc
    psi_underbar_tp1 = np.min(psi_tp1)    
    xi_underbar_tp1 = np.min(xi_tp1)
    if cubic_splines:
        vPPfunc_tp1 = solution_tp1.vPPfunc

    # Update the bounding MPCs and PDV of human wealth:
    if cubic_splines or calc_vFunc:
        thorn_R = ((R*effective_beta)**(1/rho))/R
        kappa_min_t = 1.0/(1.0 + thorn_R/solution_tp1.kappa_min)
        gothic_h_t = Gamma/R*(1.0 + solution_tp1.gothic_h)
        kappa_max_t = 1.0/(1.0 + (p_zero_income**(1/rho))*thorn_R/solution_tp1.kappa_max)
    
    # Calculate the minimum allowable value of money resources in this period
    if constrained:
        m_underbar_t = 0.0
    else:
        m_underbar_t = (solution_tp1.m_underbar - xi_underbar_tp1)*(Gamma*psi_underbar_tp1)/R

    # Define the borrowing constraint (limiting consumption function)
    constraint_t = lambda m: m - m_underbar_t

    # Find data for the unconstrained consumption function in this period
    c_temp = [0.0]  # Limiting consumption is zero as m approaches m_underbar
    m_temp = [m_underbar_t]
    if cubic_splines:
        kappa_temp = [kappa_max_t]
    if calc_vFunc:
        vQ_temp = []
        vPQ_temp = []
    for x in a_grid:
        m_t = x + m_underbar_t
        firstOrderCondition = lambda c : uP(c) - effective_beta*R*Gamma**(-rho)*np.sum(psi_tp1**(-rho)*vPfunc_tp1(R/(Gamma*psi_tp1)*(m_t-c) + xi_tp1)*prob_tp1)
        c_t = brentq(firstOrderCondition,0.001*x,0.999*x)
        c_temp.append(c_t)
        m_temp.append(m_t)
        if calc_vFunc or cubic_splines:
            m_tp1 = R/(Gamma*psi_tp1)*(m_t-c_t) + xi_tp1
        if calc_vFunc:
            V_tp1 = (psi_tp1**(1.0-rho)*Gamma**(1.0-rho))*vFunc_tp1(m_tp1)
            v_t = u(c_t) + effective_beta*np.sum(V_tp1*prob_tp1)
            vQ_temp.append(uinv(v_t)) # value transformed through inverse utility
            vPQ_temp.append(uP(c_t)*uinvP(v_t))
        if cubic_splines:
            gothicvPP = effective_beta*R*R*Gamma**(-rho-1.0)*np.sum(psi_tp1**(-rho-1.0)*vPPfunc_tp1(m_tp1)*prob_tp1)    
            dcda = gothicvPP/uPP(c_t)
            kappa_t = dcda/(dcda+1.0)
            kappa_temp.append(kappa_t)
    
    # Construct the unconstrained consumption function
    if cubic_splines:
        cFunc_t_unconstrained = Cubic1DInterpDecay(m_temp,c_temp,kappa_temp,kappa_min_t*gothic_h_t,kappa_min_t)
    else:
        cFunc_t_unconstrained = LinearInterp(m_temp,c_temp)

    # Combine the constrained and unconstrained functions into the true consumption function
    cFunc_t = ConstrainedComposite(cFunc_t_unconstrained,constraint_t)
    
    # Construct the value function if requested
    if calc_vFunc:
        k = kappa_min_t**(-rho/(1-rho))
        m_list = (np.asarray(a_grid) + m_underbar_t).tolist()
        vQfunc_t = Cubic1DInterpDecay(m_list,vQ_temp,vPQ_temp,k*gothic_h_t,k)
        vFunc_t = lambda m : u(vQfunc_t(m))
        
    # Make the marginal value function and the marginal marginal value function
    vPfunc_t = lambda m : uP(cFunc_t(m))
    if cubic_splines:
        vPPfunc_t = lambda m : cFunc_t.derivative(m)*uPP(cFunc_t(m))

    # Store the results in a solution object and return it
    if cubic_splines or calc_vFunc:
        solution_t = ConsumerSolution(cFunc=cFunc_t, vPfunc=vPfunc_t, m_underbar=m_underbar_t, gothic_h=gothic_h_t, kappa_min=kappa_min_t, kappa_max=kappa_max_t)
    else:
        solution_t = ConsumerSolution(cFunc=cFunc_t, vPfunc=vPfunc_t, m_underbar=m_underbar_t)
    if calc_vFunc:
        solution_t.vFunc = vFunc_t
    if cubic_splines:
        solution_t.vPPfunc=vPPfunc_t
    #print('Solved a period with EXOG!')
    return solution_t


    

def consumptionSavingSolverENDG(solution_tp1,income_distrib,p_zero_income,survival_prob,beta,rho,R,Gamma,constrained,a_grid,calc_vFunc,cubic_splines):
    '''
    Solves a single period of a standard consumption-saving problem, representing
    the consumption function as a cubic spline interpolation if cubic_splines is
    True and as a linear interpolation if it is False.  Problem is solved using
    the method of endogenous gridpoints.

    Parameters:
    -----------
    solution_tp1: ConsumerSolution
        The solution to the following period.
    income_distrib: [[float]]
        A list containing three lists of floats, representing a discrete approximation to the income process between
        the period being solved and the one immediately following (in solution_tp1).  Order: psi, xi, probs
    p_zero_income: float
        The probability of receiving zero income in the succeeding period.
    survival_prob: float
        Probability of surviving to succeeding period.
    beta: float
        Discount factor between this period and the succeeding period.
    rho: float
        The coefficient of relative risk aversion
    R: float
        Interest factor on assets between this period and the succeeding period: w_tp1 = a_t*R
    Gamma: float
        Expected growth factor for permanent income between this period and the succeeding period.
    constrained: bool
        Whether this individual may hold net negative assets at the end of the period.
    a_grid: [float]
        A list of end-of-period asset values (post-decision states) at which to solve for optimal consumption.
    calc_vFunc: Boolean
        An indicator for whether the value function should be computed and included
        in the reported solution
    cubic_splines: Boolean
        An indicator for whether the solver should use cubic or linear interpolation
    

    Returns:
    -----------
    solution_t: ConsumerSolution
        The solution to this period's problem, obtained using the method of endogenous gridpoints.
    '''

    # Define utility and value functions
    if calc_vFunc:
        u = lambda c : utility(c,gam=rho)
        uinv = lambda u : utility_inv(u,gam=rho)
        uinvP = lambda u : utility_invP(u,gam=rho)
        vFunc_tp1 = solution_tp1.vFunc
    uP = lambda c : utilityP(c,gam=rho)
    uPP = lambda c : utilityPP(c,gam=rho)
    uPinv = lambda u : utilityP_inv(u,gam=rho)

    # Set and update values for this period
    effective_beta = beta*survival_prob
    psi_tp1 = income_distrib[1]
    xi_tp1 = income_distrib[2]
    prob_tp1 = income_distrib[0]
    vPfunc_tp1 = solution_tp1.vPfunc
    psi_underbar_tp1 = np.min(psi_tp1)    
    xi_underbar_tp1 = np.min(xi_tp1)
    if cubic_splines:
        vPPfunc_tp1 = solution_tp1.vPPfunc

    # Update the bounding MPCs and PDV of human wealth:
    if cubic_splines or calc_vFunc:
        thorn_R = ((R*effective_beta)**(1.0/rho))/R
        kappa_min_t = 1.0/(1.0 + thorn_R/solution_tp1.kappa_min)
        gothic_h_t = Gamma/R*(1.0 + solution_tp1.gothic_h)
        kappa_max_t = 1.0/(1.0 + (p_zero_income**(1/rho))*thorn_R/solution_tp1.kappa_max)
    
    # Calculate the minimum allowable value of money resources in this period
    if constrained:
        m_underbar_t = 0.0
    else:
        m_underbar_t = (solution_tp1.m_underbar - xi_underbar_tp1)*(Gamma*psi_underbar_tp1)/R

    # Define the borrowing constraint (limiting consumption function)
    constraint_t = lambda m: m - m_underbar_t

    # Find data for the unconstrained consumption function in this period
    c_temp = [0.0]  # Limiting consumption is zero as m approaches m_underbar
    m_temp = [m_underbar_t]
    if cubic_splines:
        kappa_temp = [kappa_max_t]
    a = np.asarray(a_grid) + m_underbar_t
    a_N = a.size
    shock_N = xi_tp1.size
    a_temp = np.tile(a,(shock_N,1))
    psi_temp = (np.tile(psi_tp1,(a_N,1))).transpose()
    xi_temp = (np.tile(xi_tp1,(a_N,1))).transpose()
    prob_temp = (np.tile(prob_tp1,(a_N,1))).transpose()
    m_tp1 = R/(Gamma*psi_temp)*a_temp + xi_temp
    if calc_vFunc:
        V_tp1 = (psi_temp**(1.0-rho)*Gamma**(1.0-rho))*vFunc_tp1(m_tp1)
        gothicv = effective_beta*np.sum(V_tp1*prob_temp,axis=0)
    gothicvP = effective_beta*R*Gamma**(-rho)*np.sum(psi_temp**(-rho)*vPfunc_tp1(m_tp1)*prob_temp,axis=0)
    c = uPinv(gothicvP)
    m = c + a
    c_temp += c.tolist()
    m_temp += m.tolist()
    if cubic_splines:
        gothicvPP = effective_beta*R*R*Gamma**(-rho-1.0)*np.sum(psi_temp**(-rho-1.0)*vPPfunc_tp1(m_tp1)*prob_temp,axis=0)    
        dcda = gothicvPP/uPP(c)
        kappa = dcda/(dcda+1.0)
        kappa_temp += kappa.tolist()
    
    # Construct the unconstrained consumption function
    if cubic_splines:
        cFunc_t_unconstrained = Cubic1DInterpDecay(m_temp,c_temp,kappa_temp,kappa_min_t*gothic_h_t,kappa_min_t)
    else:
        cFunc_t_unconstrained = LinearInterp(m_temp,c_temp)

    # Combine the constrained and unconstrained functions into the true consumption function
    cFunc_t = ConstrainedComposite(cFunc_t_unconstrained,constraint_t)
    
    # Construct the value function if requested
    if calc_vFunc:
        v_temp = u(np.array(c)) + gothicv
        vQ_temp = uinv(v_temp).tolist() # value transformed through inverse utility
        vPQ_temp = gothicvP*uinvP(v_temp)
        k = kappa_min_t**(-rho/(1.0-rho))
        vQfunc_t = Cubic1DInterpDecay(m,vQ_temp,vPQ_temp,k*gothic_h_t,k)
        vFunc_t = lambda m : u(vQfunc_t(m))
        
    # Make the marginal value function and the marginal marginal value function
    vPfunc_t = lambda m : uP(cFunc_t(m))
    if cubic_splines:
        vPPfunc_t = lambda m : cFunc_t.derivative(m)*uPP(cFunc_t(m))

    # Store the results in a solution object and return it
    if cubic_splines or calc_vFunc:
        solution_t = ConsumerSolution(cFunc=cFunc_t, vPfunc=vPfunc_t, m_underbar=m_underbar_t, gothic_h=gothic_h_t, kappa_min=kappa_min_t, kappa_max=kappa_max_t)
    else:
        solution_t = ConsumerSolution(cFunc=cFunc_t, vPfunc=vPfunc_t, m_underbar=m_underbar_t)
    if calc_vFunc:
        solution_t.vFunc = vFunc_t
    if cubic_splines:
        solution_t.vPPfunc=vPPfunc_t
    #print('Solved a period with ENDG!')
    return solution_t




# ============================================================================
# == A class for representing types of consumer agents (and things they do) ==
# ============================================================================

class ConsumerType(AgentType):
    '''
    An agent in the consumption-saving model.  His problem is defined by a sequence
    of income distributions, survival probabilities, discount factors, and permanent
    income growth rates, as well as time invariant values for risk aversion, the
    interest rate, the grid of end-of-period assets, and he is borrowing constrained.
    '''    
    
    # Define some universal values for all consumer types
    cFunc_terminal_ = Cubic1DInterpDecay([0.0, 1.0],[0.0, 1.0],[1.0, 1.0],0,1)
    vFunc_terminal_ = LinearInterp([0.0, 1.0],[0.0,0.0])
    constraint_terminal_ = lambda x: x
    solution_terminal_ = ConsumerSolution(cFunc=ConstrainedComposite(cFunc_terminal_,constraint_terminal_), vFunc = vFunc_terminal_, m_underbar=0.0, gothic_h=0.0, kappa_min=1.0, kappa_max=1.0)
    time_vary_ = ['survival_prob','beta','Gamma']
    time_inv_ = ['rho','R','a_grid','constrained','calc_vFunc','cubic_splines']
    
    def __init__(self,cycles=1,time_flow=True,**kwds):
        '''
        Instantiate a new ConsumerType with given data.
        '''       
        # Initialize a basic AgentType
        AgentType.__init__(self,solution_terminal=deepcopy(ConsumerType.solution_terminal_),cycles=cycles,time_flow=time_flow,pseudo_terminal=False,**kwds)

        # Add consumer-type specific objects, copying to create independent versions
        self.time_vary = deepcopy(ConsumerType.time_vary_)
        self.time_inv = deepcopy(ConsumerType.time_inv_)
        self.solveAPeriod = consumptionSavingSolverENDG # this can be swapped for consumptionSavingSolverEXOG or another solver
        self.update()

    def unpack_cFunc(self):
        '''
        "Unpacks" the consumption functions into their own field for easier access.
        After the model has been solved, the consumption functions reside in the
        attribute cFunc of each element of ConsumerType.solution.  This method
        creates a (time varying) attribute cFunc that contains a list of consumption
        functions.
        '''
        self.cFunc = []
        for solution_t in self.solution:
            self.cFunc.append(solution_t.cFunc)
        if not ('cFunc' in self.time_vary):
            self.time_vary.append('cFunc')
            
    def addIncomeShockPaths(self,perm_shocks,temp_shocks):
        '''
        Adds paths of simulated shocks to the agent as attributes.
        '''
        original_time = self.time_flow
        self.timeFwd()
        self.perm_shocks = perm_shocks
        self.temp_shocks = temp_shocks
        if not 'perm_shocks' in self.time_vary:
            self.time_vary.append('perm_shocks')
        if not 'temp_shocks' in self.time_vary:
            self.time_vary.append('temp_shocks')
        if not original_time:
            self.timeRev()
            
    def updateIncomeProcess(self):
        '''
        Updates this agent's income process based on his own attributes.  The
        function that generates the discrete income process can be swapped out
        for a different process.
        '''
        original_time = self.time_flow
        self.timeFwd()
        income_distrib, p_zero_income = constructLognormalIncomeProcessUnemployment(self)
        self.income_distrib = income_distrib
        self.p_zero_income = p_zero_income
        if not 'income_distrib' in self.time_vary:
            self.time_vary.append('income_distrib')
        if not 'p_zero_income' in self.time_vary:
            self.time_vary.append('p_zero_income')
        if not original_time:
            self.timeRev()
            
    def updateAssetsGrid(self):
        '''
       Updates this agent's end-of-period assets grid.
        '''
        a_grid = constructAssetsGrid(self)
        self.a_grid = a_grid
        
    def updateSolutionTerminal(self):
        '''
        Update the terminal period solution
        '''
        self.solution_terminal.vFunc = lambda m : utility(m,self.rho)
        self.solution_terminal.vPfunc = lambda m : utilityP(m,self.rho)
        self.solution_terminal.vPPfunc = lambda m: self.solution_terminal.cFunc.derivative(m)*utilityPP(m,self.rho)
        
    def update(self):
        '''
        Update the income process, the assets grid, and the terminal solution.
        '''
        self.updateIncomeProcess()
        self.updateAssetsGrid()
        self.updateSolutionTerminal()
        
            
    def simulate(self,w_init,t_first,t_last,which=['w']):
        '''
        Simulate the model forward from initial conditions w_init, beginning in
        t_first and ending in t_last.
        '''
        original_time = self.time_flow
        self.timeFwd()
        if self.cycles > 0:
            cFuncs = self.cFunc[t_first:t_last]
        else:
            cFuncs = t_last*self.cFunc # This needs to be fixed for IH models
        simulated_history = simulateConsumerHistory(cFuncs, w_init, self.perm_shocks[t_first:t_last], self.temp_shocks[t_first:t_last],which)
        if not original_time:
            self.timeRev()
        return simulated_history



def simulateConsumerHistory(cFunc,w0,scriptR,theta,which):
    """
    Generates simulated consumer histories.  Agents begin with W/Y ratio of of
    w0 and follow the consumption rules in cFunc each period. Permanent and trans-
    itory shocks are provided in scriptR and theta.  Note that
    scriptR represents R*psi_{it}/Gamma_t, the "effective interest factor" for
    agent i in period t.  Further, the object of interest w is the wealth-to
    permanent-income ratio at the beginning of the period, before income is received.
    
    The histories returned by the simulator are determined by which, a list of
    strings that can include 'w', 'm', 'c', 'a', and 'kappa'.  Other strings will
    cause an error on return.  Outputs are returned in the order listed by the user.
    """
    # Determine the size of potential simulated histories
    periods_to_simulate = len(theta)
    N_agents = len(theta[0])
    
    # Initialize arrays to hold simulated histories as requested
    if 'w' in which:
        w = np.zeros([periods_to_simulate+1,N_agents]) + np.nan
        do_w = True
    else:
        do_w = False
    if 'm' in which:
        m = np.zeros([periods_to_simulate,N_agents]) + np.nan
        do_m = True
    else:
        do_m = False
    if 'c' in which:
        c = np.zeros([periods_to_simulate,N_agents]) + np.nan
        do_c = True
    else:
        do_c = False
    if 'a' in which:
        a = np.zeros([periods_to_simulate,N_agents]) + np.nan
        do_a = True
    else:
        do_a = False
    if 'kappa' in which:
        kappa = np.zeros([periods_to_simulate,N_agents]) + np.nan
        do_k = True
    else:
        do_k = False

    # Initialize the simulation
    w_t = w0
    if do_w:
        w[0,] = w_t
    
    # Run the simulation for all agents:
    for t in range(periods_to_simulate):
        m_t = w_t + theta[t]
        if do_k:
            c_t, kappa_t = cFunc[t].eval_with_derivative(m_t)
        else:
            c_t = cFunc[t](m_t)
        a_t = m_t - c_t
        w_t = scriptR[t]*a_t
        
        # Store the requested variables in the history arrays
        if do_w:
            w[t+1,] = w_t
        if do_m:
            m[t,] = m_t
        if do_c:
            c[t,] = c_t
        if do_a:
            a[t,] = a_t
        if do_k:
            kappa[t,] = kappa_t
            
    # Return the simulated histories as requested
    return_list = ''
    for var in which:
        return_list = return_list + var + ', '
    x = len(return_list)
    return_list = return_list[0:(x-2)]
    return eval(return_list)



# ==================================================================================
# = Functions for generating discrete income processes and simulated income shocks =
# ==================================================================================

def constructLognormalIncomeProcessUnemployment(parameters):
    """
    Generates a list of discrete approximations to the income process for each
    life period, from end of life to beginning of life.  Permanent shocks are mean
    one lognormally distributed with standard deviation psi_sigma[t] during the
    working life, and degenerate at 1 in the retirement period.  Transitory shocks
    are mean one lognormally distributed with a point mass at income_unemploy with
    probability p_unemploy while working; they are mean one with a point mass at
    income_unemploy_retire with probability p_unemploy_retire.  Retirement occurs
    after t=final_work_index periods of retirement.

    Note 1: All time in this function runs forward, from t=0 to t=T
    
    Note 2: All parameters are passed as attributes of the input parameters.

    Parameters:
    -----------
    psi_sigma:    [float]
        Array of standard deviations in _permanent_ income uncertainty during
        the agent's life.
    psi_N:      int
        The number of approximation points to be used in the equiprobable
        discrete approximation to the permanent income shock distribution.
    xi_sigma      [float]
        Array of standard deviations in _temporary_ income uncertainty during
        the agent's life.
    xi_N:       int
        The number of approximation points to be used in the equiprobable
        discrete approximation to the permanent income shock distribution.
    p_unemploy:             float
        The probability of becoming unemployed
    p_unemploy_retire:      float
        The probability of not receiving typical retirement income in any retired period
    T_retire:       int
        The index value i equal to the final working period in the agent's life.
        If T_retire <= 0 then there is no retirement.
    income_unemploy:         float
        Income received when unemployed. Often zero.
    income_unemploy_retire:  float
        Income received while "unemployed" when retired. Often zero.
    T_total:       int
        Total number of non-terminal periods in this consumer's life.

    Returns
    =======
    income_distrib:  [income distribution]
        Each element contains the joint distribution of permanent and transitory
        income shocks, as a set of vectors: psi_shock, xi_shock, and pmf. The
        first two are the points in the joint state space, and final vector is
        the joint pmf over those points. For example,
               psi_shock[20], xi_shock[20], and pmf[20]
        refers to the (psi, xi) point indexed by 20, with probability p = pmf[20].
    p_zero_income: [float]
        A list of probabilities of receiving exactly zero income in each period.

    """
    # Unpack the parameters from the input
    psi_sigma = parameters.psi_sigma
    psi_N = parameters.psi_N
    xi_sigma = parameters.xi_sigma
    xi_N = parameters.xi_N
    T_total = parameters.T_total
    T_retire = parameters.T_retire
    p_unemploy = parameters.p_unemploy
    income_unemploy = parameters.income_unemploy
    p_unemploy_retire = parameters.p_unemploy_retire        
    income_unemploy_retire = parameters.income_unemploy_retire
    
    income_distrib = [] # Discrete approximation to income process
    p_zero_income = [] # Probability of zero income in each period of life

    # Fill out a simple discrete RV for retirement, with value 1.0 (mean of shocks)
    # in normal times; value 0.0 in "unemployment" times with small prob.
    if T_retire > 0:
        if p_unemploy_retire > 0:
            retire_perm_income_values = np.array([1.0, 1.0])    # Permanent income is deterministic in retirement (2 states for temp income shocks)
            retire_income_values = np.array([income_unemploy_retire, (1.0-p_unemploy_retire*income_unemploy_retire)/(1.0-p_unemploy_retire)])
            retire_income_probs = np.array([p_unemploy_retire, 1.0-p_unemploy_retire])
        else:
            retire_perm_income_values = np.array([1.0])
            retire_income_values = np.array([1.0])
            retire_income_probs = np.array([1.0])
        income_dist_retire = [retire_income_probs,retire_perm_income_values,retire_income_values]

    # Loop to fill in the list of income_distrib random variables.
    for t in range(T_total): # Iterate over all periods, counting forward

        if T_retire > 0 and t >= T_retire:
            # Then we are in the "retirement period" and add a retirement income object.
            income_distrib.append(deepcopy(income_dist_retire))
            if income_unemploy_retire == 0:
                p_zero_income.append(p_unemploy_retire)
            else:
                p_zero_income.append(0)
        else:
            # We are in the "working life" periods.
            temp_xi_dist = calculateMeanOneLognormalDiscreteApprox(N=xi_N, sigma=xi_sigma[t])
            if p_unemploy > 0:
                temp_xi_dist = addDiscreteOutcomeConstantMean(temp_xi_dist, p=p_unemploy, x=income_unemploy)
            temp_psi_dist = calculateMeanOneLognormalDiscreteApprox(N=psi_N, sigma=psi_sigma[t])
            income_distrib.append(createFlatStateSpaceFromIndepDiscreteProbs(temp_psi_dist, temp_xi_dist))
            if income_unemploy == 0:
                p_zero_income.append(p_unemploy)
            else:
                p_zero_income.append(0)

    return income_distrib, p_zero_income
    
    
    
def constructLognormalIncomeProcessUnemploymentFailure(parameters):
    """
    Generates a list of discrete approximations to the income process for each
    life period, from end of life to beginning of life.  The process is identical
    to constructLognormalIncomeProcessUnemployment but for a very tiny possibility
    that unemployment benefits are not provided.

    Parameters:
    -----------
    psi_sigma:    [float]
        Array of standard deviations in _permanent_ income uncertainty during
        the agent's life.
    psi_N:      int
        The number of approximation points to be used in the equiprobable
        discrete approximation to the permanent income shock distribution.
    xi_sigma      [float]
        Array of standard deviations in _temporary_ income uncertainty during
        the agent's life.
    xi_N:       int
        The number of approximation points to be used in the equiprobable
        discrete approximation to the permanent income shock distribution.
    p_unemploy:             float
        The probability of becoming unemployed
    p_unemploy_retire:      float
        The probability of not receiving typical retirement income in any retired period
    T_retire:       int
        The index value i equal to the final working period in the agent's life.
        If T_retire <= 0 then there is no retirement.
    income_unemploy:         float
        Income received when unemployed. Often zero.
    income_unemploy_retire:  float
        Income received while "unemployed" when retired. Often zero.
    T_total:       int
        Total number of non-terminal periods in this consumer's life.

    Returns
    =======
    income_distrib:  [income distribution]
        Each element contains the joint distribution of permanent and transitory
        income shocks, as a set of vectors: psi_shock, xi_shock, and pmf. The
        first two are the points in the joint state space, and final vector is
        the joint pmf over those points. For example,
               psi_shock[20], xi_shock[20], and pmf[20]
        refers to the (psi, xi) point indexed by 20, with probability p = pmf[20].
    p_zero_income: [float]
        A list of probabilities of receiving exactly zero income in each period.

    """
    # Unpack the parameters from the input
    psi_sigma = parameters.psi_sigma
    psi_N = parameters.psi_N
    xi_sigma = parameters.xi_sigma
    xi_N = parameters.xi_N
    T_total = parameters.T_total
    p_unemploy = parameters.p_unemploy
    p_unemploy_retire = parameters.p_unemploy_retire
    T_retire = parameters.T_retire
    income_unemploy = parameters.income_unemploy
    income_unemploy_retire = parameters.income_unemploy_retire

    # Set a small possibility of unemployment benefit failure
    p_fail = 0.01
    
    income_distrib = [] # Discrete approximation to income process
    p_zero_income = [] # Probability of zero income in each period of life

    # Fill out a simple discrete RV for retirement, with value 1.0 (mean of shocks)
    # in normal times; value 0.0 in "unemployment" times with small prob.
    if T_retire > 0:
        if p_unemploy_retire > 0:
            retire_perm_income_values = np.array([1.0, 1.0])    # Permanent income is deterministic in retirement (2 states for temp income shocks)
            retire_income_values = np.array([income_unemploy_retire, (1.0-p_unemploy_retire*income_unemploy_retire)/(1.0-p_unemploy_retire)])
            retire_income_probs = np.array([p_unemploy_retire, 1.0-p_unemploy_retire])
            if income_unemploy_retire > 0:
                temp_dist = addDiscreteOutcomeConstantMean([retire_income_values, retire_income_probs], p=(p_fail*p_unemploy_retire), x=0)
                retire_income_values = temp_dist[0]
                retire_income_probs = temp_dist[1]
                retire_perm_income_values = np.array([1.0, 1.0, 1.0])
        else:
            retire_perm_income_values = np.array([1.0])
            retire_income_values = np.array([1.0])
            retire_income_probs = np.array([1.0])
        income_dist_retire = [retire_income_probs,retire_perm_income_values,retire_income_values]


    # Loop to fill in the list of income_distrib random variables.
    for t in range(T_total): # Iterate over all periods, counting forward

        if T_retire > 0 and t >= T_retire:
            # Then we are in the "retirement period" and add a retirement income object.
            income_distrib.append(deepcopy(income_dist_retire))
            if income_unemploy_retire == 0:
                p_zero_income.append(p_unemploy_retire)
            else:
                p_zero_income.append(p_fail*p_unemploy_retire)
        else:
            # We are in the "working life" periods.
            temp_xi_dist = calculateMeanOneLognormalDiscreteApprox(N=xi_N, sigma=xi_sigma[t])
            if p_unemploy > 0:
                temp_xi_dist = addDiscreteOutcomeConstantMean(temp_xi_dist, p=p_unemploy, x=income_unemploy)
                if income_unemploy > 0:
                    temp_xi_dist = addDiscreteOutcomeConstantMean(temp_xi_dist, p=(p_unemploy*p_fail), x=0)
            temp_psi_dist = calculateMeanOneLognormalDiscreteApprox(N=psi_N, sigma=psi_sigma[t])
            income_distrib.append(createFlatStateSpaceFromIndepDiscreteProbs(temp_psi_dist, temp_xi_dist))
            if p_unemploy > 0:
                if income_unemploy == 0:
                    p_zero_income.append(p_unemploy)
                else:
                    p_zero_income.append(p_unemploy*p_fail)
            else:
                p_zero_income.append(0)

    return income_distrib, p_zero_income


def applyFlatIncomeTax(income_distrib,tax_rate,T_retire,unemployed_indices=[],transitory_index=2):
    '''
    Applies a flat income tax rate to all employed income states during the working
    period of life (those before T_retire).  Time runs forward in this function.
    
    Parameters:
    -------------
    income_distrib : [income distributions]
        The discrete approximation to the income distribution in each time period.
    tax_rate : float
        A flat income tax rate to be applied to all employed income.
    T_retire : int
        The time index after which the agent retires.
    unemployed_indices : [int]
        Indices of transitory shocks that represent unemployment states (no tax).
    transitory_index : int
        The index of each element of income_distrib representing transitory shocks.
        
    Returns:
    ------------
    income_distrib_new : [income distributions]
        The updated income distributions, after applying the tax.
    '''
    income_distrib_new = deepcopy(income_distrib)
    i = transitory_index
    for t in range(len(income_distrib)):
        if t < T_retire:
            for j in range((income_distrib[t][i]).size):
                if j not in unemployed_indices:
                    income_distrib_new[t][i][j] = income_distrib[t][i][j]*(1-tax_rate)
    return income_distrib_new
   
    
    
def generateIncomeShockHistoryLognormalUnemployment(parameters):
    '''
    Creates arrays of permanent and transitory income shocks for Nagents simulated
    consumers for the entire duration of the lifecycle.  All inputs are assumed
    to be given in ordinary chronological order, from terminal period to t=0.
    Output is also returned in ordinary chronological order.
    
    Arguments:
    ----------
    psi_sigma : [float]
        Permanent income standard deviations for the consumer by age.
    xi_sigma : [float]
        Transitory income standard devisions for the consumer by age.
    Gamma : [float]
        Permanent income growth rates for the consumer by age.
    R : [float]
        The time-invariant interest factor
    p_unemploy : float
        The probability of becoming unemployed
    p_unemploy_retire : float
        The probability of not receiving typical retirement income in any retired period
    T_retire : int
        The index value for final working period in the agent's life.
        If T_retire <= 0 then there is no retirement.
    income_unemploy : float
        Income received when unemployed. Often zero.
    income_unemploy_retire : float
        Income received while "unemployed" when retired. Often zero.
    Nagents : int
        The number of consumers to generate shocks for.
    tax_rate : float
        An income tax rate applied to employed income.
    psi_seed : int
        Seed for random number generator, permanent income shocks.
    xi_seed : int
        Seed for random number generator, temporary income shocks.
    unemp_seed : int
        Seed for random number generator, unemployment shocks.

    Returns:
    ----------
    scriptR_history : np.array
        A total_periods x Nagents array of permanent income shocks.  Each element
        is a value representing R/(psi_{it}*Gamma_t), so that w_{t+1} = scriptR_{it}*a_t
    xi_history : np.array
        A total_periods x Nagents array of transitory income shocks.
    '''
    # Unpack the parameters
    psi_sigma = parameters.psi_sigma
    xi_sigma = parameters.xi_sigma
    Gamma = parameters.Gamma
    R = parameters.R
    p_unemploy = parameters.p_unemploy
    p_unemploy_retire = parameters.p_unemploy_retire
    income_unemploy = parameters.income_unemploy
    income_unemploy_retire = parameters.income_unemploy_retire
    T_retire = parameters.T_retire
    Nagents = parameters.Nagents
    psi_seed = parameters.psi_seed
    xi_seed = parameters.xi_seed
    unemp_seed = parameters.unemp_seed
    tax_rate = parameters.tax_rate

    # Truncate the lifecycle vectors to the working life
    psi_sigma_working = psi_sigma[0:T_retire]
    xi_sigma_working = xi_sigma[0:T_retire]
    Gamma_working = Gamma[0:T_retire]
    Gamma_retire = Gamma[T_retire:]
    working_periods = len(Gamma_working) + 1
    retired_periods = len(Gamma_retire)
    
    # Generate transitory shocks in the working period (needs one extra period)
    xi_history_working = generateMeanOneLognormalDraws(xi_sigma_working, Nagents, xi_seed)
    np.random.seed(0)
    xi_history_working.insert(0,np.random.permutation(xi_history_working[0]))
    
    # Generate permanent shocks in the working period
    scriptR_history_working = generateMeanOneLognormalDraws(psi_sigma_working, Nagents, psi_seed)
    for t in range(working_periods-1):
        scriptR_history_working[t] = R/(scriptR_history_working[t]*Gamma_working[t])

    # Generate permanent and transitory shocks for the retired period
    xi_history_retired = []
    scriptR_history_retired = []
    for t in range(retired_periods):
        xi_history_retired.append(np.ones([Nagents]))
        scriptR_history_retired.append(R*np.ones([Nagents])/Gamma_retire[t])
    scriptR_history_retired.append(R*np.ones([Nagents]))
    
    # Generate draws of unemployment
    p_unemploy_life = [p_unemploy]*working_periods + [p_unemploy_retire]*retired_periods
    income_unemploy_life = [income_unemploy]*working_periods + [income_unemploy_retire]*retired_periods
    unemp_rescale_life = [(1-tax_rate)*(1-p_unemploy*income_unemploy)/(1-p_unemploy)]*working_periods + [(1-p_unemploy_retire*income_unemploy_retire)/(1-p_unemploy_retire)]*retired_periods
    unemployment_history = generateBernoulliDraws(p_unemploy_life,Nagents,unemp_seed)   
    
    # Combine working and retired histories and apply unemployment
    xi_history = xi_history_working + xi_history_retired
    scriptR_history = scriptR_history_working + scriptR_history_retired
    for t in range(len(xi_history)):
        xi_history[t] = xi_history[t]*unemp_rescale_life[t]
        xi_history[t][unemployment_history[t]] = income_unemploy_life[t]
    
    return scriptR_history, xi_history
    
    
def generateIncomeShockHistoryInfiniteSimple(parameters):
    '''
    Creates arrays of permanent and transitory income shocks for Nagents simulated
    consumers for T identical infinite horizon periods.
    
    Arguments:
    ----------
    psi_sigma : float
        Permanent income standard deviation for the consumer.
    xi_sigma : float
        Transitory income standard deviation for the consumer.
    Gamma : float
        Permanent income growth rate for the consumer.
    R : float
        The time-invariant interest factor
    p_unemploy : float
        The probability of becoming unemployed
    income_unemploy : float
        Income received when unemployed. Often zero.
    Nagents : int
        The number of consumers to generate shocks for.
    psi_seed : int
        Seed for random number generator, permanent income shocks.
    xi_seed : int
        Seed for random number generator, temporary income shocks.
    unemp_seed : int
        Seed for random number generator, unemployment shocks.
    sim_periods : int
        Number of periods of shocks to generate.
    
    Returns:
    ----------
    scriptR_history : np.array
        A sim_periods x Nagents array of permanent income shocks.  Each element
        is a value representing R*psi_{it}/Gamma_t, so that w_{t+1} = scriptR_{it}*a_t
    xi_history : np.array
        A sim_periods x Nagents array of transitory income shocks.
    '''
    # Unpack the parameters
    psi_sigma = parameters.psi_sigma
    xi_sigma = parameters.xi_sigma
    Gamma = parameters.Gamma
    R = parameters.R
    p_unemploy = parameters.p_unemploy
    income_unemploy = parameters.income_unemploy
    Nagents = parameters.Nagents
    psi_seed = parameters.psi_seed
    xi_seed = parameters.xi_seed
    unemp_seed = parameters.unemp_seed
    sim_periods = parameters.sim_periods
    
    xi_history = generateMeanOneLognormalDraws(sim_periods*xi_sigma, Nagents, xi_seed)
    unemployment_history = generateBernoulliDraws(sim_periods*[p_unemploy],Nagents,unemp_seed)
    scriptR_history = generateMeanOneLognormalDraws(sim_periods*psi_sigma, Nagents, psi_seed)
    for t in range(sim_periods):
        scriptR_history[t] = R/(scriptR_history[t]*Gamma)
        xi_history[t] = xi_history[t]*(1-p_unemploy*income_unemploy)/(1-p_unemploy)
        xi_history[t][unemployment_history[t]] = income_unemploy
        
    return scriptR_history, xi_history

# =======================================================
# ================ Other useful functions ===============
# =======================================================

def constructAssetsGrid(parameters):
    '''
    Constructs the grid of post-decision states, representing end-of-period assets.

    All parameters are passed as attributes of the single input parameters.  The
    input can be an instance of a ConsumerType, or a custom Parameters class.    
    
    Parameters:
    -----------
    a_min:                  float
        Minimum value for the a-grid
    a_max:                  float
        Maximum value for the a-grid
    a_size:                 int
        Size of the a-grid
    a_extra:                [float]
        Extra values for the a-grid.
    grid_type:              string
        String indicating the type of grid. "linear" or "exp_mult"
    exp_nest:               int
        Level of nesting for the exponentially spaced grid
        
    Returns:
    ----------
    a_grid:     np.ndarray
        Base array of values for the post-decision-state grid.
    '''
    # Unpack the parameters
    a_min = parameters.a_min
    a_max = parameters.a_max
    a_size = parameters.a_size
    a_extra = parameters.a_extra
    grid_type = 'exp_mult'
    exp_nest = parameters.exp_nest
    
    # Set up post decision state grid:
    a_grid = None
    if grid_type == "linear":
        a_grid = np.linspace(a_min, a_max, a_size)
    elif grid_type == "exp_mult":
        a_grid = setupGridsExpMult(ming=a_min, maxg=a_max, ng=a_size, timestonest=exp_nest)
    else:
        raise Exception, "grid_type not recognized in __init__. Please ensure grid_type is 'linear' or 'exp_mult'"

    # Add in additional points for the grid:
    for a in a_extra:
        if (a is not None):
            if a not in a_grid:
                j = a_grid.searchsorted(a)
                a_grid = np.insert(a_grid, j, a)

    return a_grid