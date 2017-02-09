'''
Demonstrates an example estimation of microeconomic dynamic stochastic optimization
problem, as described in Section 9 of Chris Carroll's SolvingMicroDSOPs.pdf notes.
The estimation attempts to match the age-conditional wealth profile of simulated
consumers to the median wealth holdings of seven age groups in the 2004 SCF by
varying only two parameters: the coefficient of relative risk aversion and a scaling
factor for an age-varying sequence of discount factors.  The estimation uses a
consumption-saving model with idiosyncratic shocks to permanent and transitory
income as defined in ConsIndShockModel.

'''
# Import the HARK library.  The assumption is that this code is in a folder
# contained in the HARK folder. 
import sys 
import os
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('../ConsumptionSaving'))

import EstimationParameters_temp as Params      # Parameters for the consumer type and the estimation
import ConsIndShockModel as Model               # The consumption-saving micro model
import SetupSCFdata as Data                     # SCF 2004 data on household wealth
from HARKsimulation import drawDiscrete         # Method for sampling from a discrete distribution
from HARKestimation import minimizeNelderMead, bootstrapSampleFromData # Estimation methods
import numpy as np                              # Numeric Python
import pylab                                    # Python reproductions of some Matlab functions
from time import time                           # Timing utility

# NOTE: now import a couple helper functions: 
from quantiles_library import weighted_quantile, goh, quantiles, quantile_names, float_quantiles
from copy import deepcopy


# Set booleans to determine which tasks should be done
estimate_model = True             # Whether to estimate the model
compute_standard_errors = False   # Whether to get standard errors via bootstrap
make_contour_plot = False         # Whether to make a contour map of the objective function

#=====================================================
# Define objects and functions used for the estimation
#=====================================================

class TempConsumerType(Model.IndShockConsumerType):
    '''
    A very lightly edited version of IndShockConsumerType.  Uses an alternate method of making new
    consumers and specifies DiscFac as being age-dependent.  Called "temp" because only used here.
    '''
    def __init__(self,cycles=1,time_flow=True,**kwds):
        '''
        Make a new consumer type.
        
        Parameters
        ----------
        cycles : int
            Number of times the sequence of periods should be solved.
        time_flow : boolean
            Whether time is currently "flowing" forward for this instance.
        
        Returns
        -------
        None
        '''       
        # Initialize a basic AgentType
        Model.IndShockConsumerType.__init__(self,cycles=cycles,time_flow=time_flow,**kwds)
        self.addToTimeVary('DiscFac') # This estimation uses age-varying discount factors as
        self.delFromTimeInv('DiscFac')# estimated by Cagetti (2003), so switch from time_inv to time_vary
        
    def simBirth(self,which_agents):
        '''
        Alternate method for simulating initial states for simulated agents, drawing from a finite
        distribution.  Used to overwrite IndShockConsumerType.simBirth, which uses lognormal distributions.
        
        Parameters
        ----------
        which_agents : np.array(Bool)
            Boolean array of size self.AgentCount indicating which agents should be "born".
        
        Returns
        -------
        None
        '''
        # Get and store states for newly born agents
        self.aNrmNow[which_agents] = self.aNrmInit[which_agents] # Take directly from pre-specified distribution
        self.pLvlNow[which_agents] = 1.0 # No variation in permanent income needed
        self.t_age[which_agents]   = 0 # How many periods since each agent was born
        self.t_cycle[which_agents] = 0 # Which period of the cycle each agents is currently in
        return None


# Make a lifecycle consumer to be used for estimation, including simulated shocks (plus an initial distribution of wealth)
EstimationAgent = TempConsumerType(**Params.init_consumer_objects)   # Make a TempConsumerType for estimation
EstimationAgent(T_sim = EstimationAgent.T_cycle+1)                   # Set the number of periods to simulate
EstimationAgent.track_vars = ['bNrmNow']                             # Choose to track bank balances as wealth
EstimationAgent.aNrmInit = drawDiscrete(N=Params.num_agents,
                                      P=Params.initial_wealth_income_ratio_probs,
                                      X=Params.initial_wealth_income_ratio_vals,                                      
                                      seed=Params.seed)              # Draw initial assets for each consumer
EstimationAgent.makeShockHistory()


# ==============================================================================
# Let's construct the empirical quantiles and weights needed
# ==============================================================================
empirical_data = Data.w_to_y_data
empirical_weights = Data.empirical_weights
empirical_groups = Data.empirical_groups

unique_group_numbers = np.unique(empirical_groups)
count_per_group = [np.sum(empirical_weights[empirical_groups == n]) for n in unique_group_numbers]
total_count_obs = np.sum(count_per_group)

N_values_per_group = {}
N_values_per_group.update(zip(unique_group_numbers, count_per_group))

weights_per_group = N_values_per_group.copy()
normalize_group_weights = True
if normalize_group_weights:
    for key, val in N_values_per_group.iteritems():
        weights_per_group[key] = val / float(total_count_obs)

phi_empirical_pool = []
phi_empirical_pool_weight = []
for n in unique_group_numbers:
    
    # Select data in this group again:
    group_index = empirical_groups == n
    temp_data = empirical_data[ group_index ]
    temp_weights = empirical_weights[ group_index ]

    assert len(temp_data) == len(temp_weights)
    print("For age group, "+str(n)+ " N obs = "+ str(len(temp_data)))
    
    # Find the quantiles for each age group:
    quantile_results = weighted_quantile(values=temp_data,
                                         quantiles=float_quantiles, 
                                         sample_weight=temp_weights)
    # Save all values in a dict:
    temp_quantiles = {}
    temp_quantiles.update( zip(quantile_names, quantile_results) )
    
    # Calculate the Phi values:
    temp_phi_vals = goh(temp_quantiles)

    phi_empirical_pool += [x for x in temp_phi_vals]
    phi_empirical_pool_weight += [weights_per_group[n]] * len(temp_phi_vals)
    
    # delete things:
    temp_quantiles = None


# Define the objective function for the simulated method of moments estimation
def smmObjectiveFxn(DiscFacAdj, CRRA,
                    return_phi = False,
                    agent = EstimationAgent,
                    DiscFacAdj_bound = Params.DiscFacAdj_bound,
                    CRRA_bound = Params.CRRA_bound,
                    phi_empirical=phi_empirical_pool,
                    phi_empirical_weight=phi_empirical_pool_weight,
                    map_simulated_to_empirical_cohorts = Data.simulation_map_cohorts_to_age_indices,
                    quantile_names=quantile_names, 
                    float_quantiles=float_quantiles):
    '''
    The objective function for the SMM estimation.  Given values of discount factor
    adjuster DiscFacAdj, coeffecient of relative risk aversion CRRA, a base consumer
    agent type, empirical data, and calibrated parameters, this function calculates
    the weighted distance between data and the simulated wealth-to-permanent
    income ratio.

    Steps:
        a) solve for consumption functions for (DiscFacAdj, CRRA)
        b) simulate wealth holdings for many consumers over time
        c) sum distances between empirical data and simulated medians within
            seven age groupings
            
    Parameters
    ----------
    DiscFacAdj : float
        An adjustment factor to a given age-varying sequence of discount factors.
        I.e. DiscFac[t] = DiscFacAdj*DiscFac_timevary[t].
    CRRA : float
        Coefficient of relative risk aversion.
    agent : ConsumerType
        The consumer type to be used in the estimation, with all necessary para-
        meters defined except the discount factor and CRRA.
    DiscFacAdj_bound : (float,float)
        Lower and upper bounds on DiscFacAdj; if outside these bounds, the function
        simply returns a "penalty value".
    DiscFacAdj_bound : (float,float)
        Lower and upper bounds on CRRA; if outside these bounds, the function
        simply returns a "penalty value".
    empirical_data : np.array
        Array of wealth-to-permanent-income ratios in the data.
    empirical_weights : np.array
        Weights for each observation in empirical_data.
    empirical_groups : np.array
        Array of integers listing the age group for each observation in empirical_data.
    map_simulated_to_empirical_cohorts : [np.array]
        List of arrays of "simulation ages" for each age grouping.  E.g. if the
        0th element is [1,2,3,4,5], then these time indices from the simulation
        correspond to the 0th empirical age group.
        
    Returns
    -------
    distance_sum : float
        Sum of distances between empirical data observations and the corresponding
        median wealth-to-permanent-income ratio in the simulation.
    '''   
    original_time_flow = agent.time_flow
    agent.timeFwd() # Make sure time is flowing forward for the agent
    
    # A quick check to make sure that the parameter values are within bounds.
    # Far flung falues of DiscFacAdj or CRRA might cause an error during solution or 
    # simulation, so the objective function doesn't even bother with them.
    if DiscFacAdj < DiscFacAdj_bound[0] or DiscFacAdj > DiscFacAdj_bound[1] or CRRA < CRRA_bound[0] or CRRA > CRRA_bound[1]:
        return 1e30
        
    # Update the agent with a new path of DiscFac based on this DiscFacAdj (and a new CRRA)
    agent(DiscFac = [b*DiscFacAdj for b in Params.DiscFac_timevary], CRRA = CRRA)
    
    # Solve the model for these parameters, then simulate wealth data
    agent.solve()        # Solve the microeconomic model
    agent.unpackcFunc() # "Unpack" the consumption function for convenient access
    max_sim_age = max([max(ages) for ages in map_simulated_to_empirical_cohorts])+1
    agent.initializeSim()                     # Initialize the simulation by clearing histories, resetting initial values
    agent.simulate(max_sim_age)               # Simulate histories of consumption and wealth
    sim_w_history = agent.bNrmNow_hist        # Take "wealth" to mean bank balances before receiving labor income
    
    # Find the distance between empirical data and simulated medians for each age group
    group_count = len(map_simulated_to_empirical_cohorts)

    phi_simulated_pool = []
    for g in range(group_count):

        cohort_indices = map_simulated_to_empirical_cohorts[g]
        dist = sim_w_history[cohort_indices,]

        # Find the quantiles for each age group:
        quantile_results = weighted_quantile(values=dist.flatten(),
                                             quantiles=float_quantiles, 
                                             sample_weight=None)
        # Save all values in a dict:
        temp_quantiles = {}
        temp_quantiles.update( zip(quantile_names, quantile_results) )
        
        # Calculate the Phi values:
        temp_phi_vals = goh(temp_quantiles)
        phi_simulated_pool += [x for x in temp_phi_vals]

        # delete things:
        temp_quantiles = None
        temp_phi_vals = None


    phi_diff = np.array(phi_simulated_pool) - np.array(phi_empirical)
    W_array = np.array(phi_empirical_weight)

    # NOW let's construct the values:
    Phi_T = np.atleast_2d(phi_diff)
    W = np.diag(W_array)
    Phi = Phi_T.T

    distance_sum = np.dot( Phi_T.dot(W), Phi)[0, 0]

    # Restore time to its original direction and report the result
    if not original_time_flow:
        agent.timeRev()

    if return_phi:
        return distance_sum, phi_simulated_pool, phi_empirical
    else:
        return distance_sum





# Make a single-input lambda function for use in the optimizer
smmObjectiveFxnReduced = lambda parameters_to_estimate : smmObjectiveFxn(DiscFacAdj=parameters_to_estimate[0],CRRA=parameters_to_estimate[1])

smmObjectiveFxnReduced_check_phi = lambda parameters_to_estimate : smmObjectiveFxn(DiscFacAdj=parameters_to_estimate[0],CRRA=parameters_to_estimate[1], return_phi=True)

if __name__ == '__main__':
    from time import time
    opt=False
    if opt:
        initial_guess = [Params.DiscFacAdj_start,Params.CRRA_start]
        print('Now estimating the model using Nelder-Mead from an initial guess of' + str(initial_guess) + '...')
        model_estimate = minimizeNelderMead(smmObjectiveFxnReduced,initial_guess,verbose=True)
        print('Estimated values: DiscFacAdj=' + str(model_estimate[0]) + ', CRRA=' + str(model_estimate[1]))
    else:
        # must be manual
        model_estimate = [1.00886938188, 2.7484689884]

    check_output = True
    if check_output:
        dist_sum, phi_sim_pool, phi_empiric_pool = smmObjectiveFxnReduced_check_phi(model_estimate)
        phi_sim_pool = np.reshape(phi_sim_pool, (7,4))
        phi_empiric_pool = np.reshape(phi_empiric_pool, (7,4))

        dist_sum2, phi_sim_pool2, phi_empiric_pool2 = smmObjectiveFxnReduced_check_phi( (0.997989348521, 3.2574858072) )
        phi_sim_pool2 = np.reshape(phi_sim_pool2, (7,4))
        phi_empiric_pool2 = np.reshape(phi_empiric_pool2, (7,4))


    make_contour_plot = True
    if make_contour_plot:
        grid_density = 10   # Number of parameter values in each dimension
        level_count = 100   # Number of contour levels to plot
        #DiscFacAdj_list = np.linspace(0.9,1.05,grid_density)
        #CRRA_list = np.linspace(2,8,grid_density)

        DiscFacAdj_list = np.linspace(1.002,1.007,grid_density)
        CRRA_list = np.linspace(2.7,2.8,grid_density)

        CRRA_mesh, DiscFacAdj_mesh = pylab.meshgrid(CRRA_list,DiscFacAdj_list)
        smm_obj_levels = np.empty([grid_density,grid_density])
        t0=time()
        for j in range(grid_density):
            DiscFacAdj = DiscFacAdj_list[j]
            for k in range(grid_density):
                CRRA = CRRA_list[k]
                smm_obj_levels[j,k] = smmObjectiveFxn(DiscFacAdj,CRRA) 
        t1=time()  
        print("took: ", (t1-t0)/60.0, " min")
        smm_contour = pylab.contourf(CRRA_mesh,DiscFacAdj_mesh,smm_obj_levels,level_count)
        pylab.colorbar(smm_contour)
        
        myshape=smm_obj_levels.shape
        z=np.unravel_index(np.argmin(smm_obj_levels),myshape)

        model_estimate2=[DiscFacAdj_mesh[z], CRRA_mesh[z]]

        pylab.plot(model_estimate2[1],model_estimate2[0],'*r',ms=15)
        pylab.xlabel(r'coefficient of relative risk aversion $\rho$',fontsize=14)
        pylab.ylabel(r'discount factor adjustment $\beth$',fontsize=14)
        pylab.savefig('SMMcontour7.pdf')
        pylab.savefig('SMMcontour7.png')
        pylab.show()

        print("Model est from grid:", model_estimate2) 




