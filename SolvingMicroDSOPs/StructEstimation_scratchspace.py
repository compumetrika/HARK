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
# Let's construct the emperical quantiles and weights needed
# ==============================================================================
empirical_data = Data.w_to_y_data
empirical_weights = Data.empirical_weights
empirical_groups = Data.empirical_groups

unique_group_numbers = np.unique(empirical_groups)
count_per_group = [np.sum(empirical_groups == n) for n in unique_group_numbers]
total_count_obs = np.sum(count_per_group)

N_values_per_group = {}
N_values_per_group.update(zip(unique_group_numbers, count_per_group))

weights_per_group = N_values_per_group.copy()
normalize_group_weights = False
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




# ==============================================================================
# ==============================================================================
#
# Ok, let's run a single estimation run, given a few filled-in things. 
#
# ==============================================================================
# ==============================================================================

# Set the values to estimate:
DiscFacAdj = Params.DiscFacAdj_start
CRRA = Params.CRRA_start

DiscFacAdj = 0.997989348521
CRRA = 3.2574858072

agent = EstimationAgent
DiscFacAdj_bound = Params.DiscFacAdj_bound
CRRA_bound = Params.CRRA_bound
empirical_data = Data.w_to_y_data
empirical_weights = Data.empirical_weights
empirical_groups = Data.empirical_groups
map_simulated_to_empirical_cohorts = Data.simulation_map_cohorts_to_age_indices

# Ok now set the code to run from the guts of the function: 
original_time_flow = agent.time_flow
agent.timeFwd() # Make sure time is flowing forward for the agent

# A quick check to make sure that the parameter values are within bounds.
# Far flung falues of DiscFacAdj or CRRA might cause an error during solution or 
# simulation, so the objective function doesn't even bother with them.
if DiscFacAdj < DiscFacAdj_bound[0] or DiscFacAdj > DiscFacAdj_bound[1] or CRRA < CRRA_bound[0] or CRRA > CRRA_bound[1]:
    RETURN_VALUE = 1e30
    
# Update the agent with a new path of DiscFac based on this DiscFacAdj (and a new CRRA)
agent(DiscFac = [b*DiscFacAdj for b in Params.DiscFac_timevary], CRRA = CRRA)


# Solve the model for these parameters, then simulate wealth data
agent.solve()                               # Solve the microeconomic model
agent.unpackcFunc()                         # "Unpack" the consumption function for convenient access
max_sim_age = max([max(ages) for ages in map_simulated_to_empirical_cohorts])+1
agent.initializeSim()                       # Initialize the simulation by clearing histories, resetting initial values
agent.simulate(max_sim_age)                 # Simulate histories of consumption and wealth
sim_w_history = agent.bNrmNow_hist          # Take "wealth" to mean bank balances before receiving labor income


# NOW CREATE A TIME SERIES LIKE THE OTHERS.
# NOTE: WEIGHTS ARE ALL JUST 1.0
#
# So for each age range, just pool all values into a single vector, give all the same age:
#
# Set up the number of values in each age_group:
N_values_per_group = []
for n in np.unique(empirical_groups):
    N_values_per_group.append( np.sum(empirical_groups == n) )

weights_per_group = np.array(N_values_per_group) / float(sum(N_values_per_group))

N_quantile_functions = 4
group_count = len(map_simulated_to_empirical_cohorts)


simulated_data_list = []
simulated_weights_list = []
simulated_groups_list = []

list_of_simulated_quantiles = []
phi_simulated_list = []
phi_simulated_pool = []
phi_simulated_pool_myweight = []
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

    list_of_simulated_quantiles.append(deepcopy(quantile_results))
    
    # Calculate the Phi values:
    temp_phi_vals = goh(temp_quantiles)
    phi_simulated_list.append( deepcopy(temp_phi_vals) )

    phi_simulated_pool += [x for x in temp_phi_vals]
    phi_simulated_pool_myweight += [weights_per_group[g]] * len(temp_phi_vals)

    # delete things:
    temp_quantiles = None
    temp_phi_vals = None
# NOW let's create the various things needed:
#weighted_quantile, goh, quantiles, quantile_names, float_quantiles

# Now let's create a set of vectors: 
#list_of_quantiles = []
#phi_list = []
#for n in unique_ages:


# ==============================================================================
# OK, NOW Let's find the empirical set of equivalent distributions
# ==============================================================================
list_of_empirical_quantiles = []
phi_empirical_list = []
phi_empirical_pool = []
#phi_empirical_pool_myweight = []
for n in np.unique(empirical_groups):
    
    # Select data in this group again:
    #temp_data = df['wealth_income_ratio'][ df['age_group'] == n ]
    #temp_weights = df['weight'][ df['age_group'] == n ]
    temp_data = empirical_data[ empirical_groups == n ]
    temp_weights = empirical_weights[ empirical_groups == n ]

    assert len(temp_data) == len(temp_weights)
    print("For age group, "+str(n)+ " N obs = "+ str(len(temp_data)))
    
    # Find the quantiles for each age group:
    quantile_results = weighted_quantile(values=temp_data,
                                         quantiles=float_quantiles, 
                                         sample_weight=temp_weights)
    # Save all values in a dict:
    temp_quantiles = {}
    for q, q_result in zip(quantile_names, quantile_results):
        temp_quantiles[q] = q_result
    
    list_of_empirical_quantiles.append(deepcopy(quantile_results))
    
    # Calculate the Phi values:
    temp_phi_vals = goh(temp_quantiles)
    phi_empirical_list.append( deepcopy(temp_phi_vals) )

    phi_empirical_pool += [x for x in temp_phi_vals]
    #phi_empirical_pool_myweight += [weights_per_group[g]] * len(temp_phi_vals)
    
    # delete things:
    temp_quantiles = None

phi_diff = np.array(phi_simulated_pool) - np.array(phi_empirical_pool)
W_array = np.array(phi_simulated_pool_myweight)

# NOW let's construct the values:
Phi_T = np.atleast_2d(phi_diff)
W = np.diag(W_array)
Phi = Phi_T.T

distance_sum2 = np.dot( Phi_T.dot(W), Phi)[0, 0]


print("distance_sum2:",distance_sum2)
















# Find the distance between empirical data and simulated medians for each age group
distance_sum = 0
for g in range(group_count):
    cohort_indices = map_simulated_to_empirical_cohorts[g] # The simulated time indices corresponding to this age group
    sim_median = np.median(sim_w_history[cohort_indices,]) # The median of simulated wealth-to-income for this age group
    group_indices = empirical_groups == (g+1) # groups are numbered from 1
    distance_sum += np.dot(np.abs(empirical_data[group_indices] - sim_median),empirical_weights[group_indices]) # Weighted distance from each empirical observation to the simulated median for this age group

# Restore time to its original direction and report the result
if not original_time_flow:
    agent.timeRev()  

RETURN_VALUE = distance_sum

print "HEY HAVE YOU SEEN THIS DISTANCE SUM? SWEET:\n", RETURN_VALUE



