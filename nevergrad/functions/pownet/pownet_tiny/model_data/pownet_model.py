# coding: utf-8
from __future__ import division # convert int or long division arguments to floating point values before division
from pyomo.environ import *
from pyomo.opt import SolverFactory
import itertools

gd_nodes = ['GS1','GS2','GS3','GS5','GS7','KPCM','KPT','SHV','SRP'] ##Dispatchables with demand
gn_nodes = ['STH','Thai','Viet'] ##Dispatchables without demand

g_nodes = gd_nodes + gn_nodes
print ('Gen_Nodes:',len(g_nodes))


model = AbstractModel()


######=================================================########
######               Segment B.1                       ########
######=================================================########

## string indentifiers for the set of generators (in the order of g_nodes list)
model.GD1Gens =  Set()
model.GD2Gens =  Set()
model.GD3Gens =  Set()
model.GD4Gens =  Set()
model.GD5Gens =  Set()
model.GD6Gens =  Set()
model.GD7Gens =  Set()
model.GD8Gens =  Set()
model.GD9Gens =  Set()

model.GN1Gens =  Set()
model.GN2Gens =  Set()
model.GN3Gens =  Set()


model.Generators = model.GD1Gens | model.GD2Gens | model.GD3Gens | model.GD4Gens | \
                   model.GD5Gens | model.GD6Gens | model.GD7Gens | model.GD8Gens | \
                   model.GD9Gens | model.GN1Gens | model.GN2Gens | model.GN3Gens
                   

### Generators by fuel-type
model.Coal_st = Set()
model.Oil_ic = Set()
model.Oil_st = Set()
model.Imp_Viet = Set()
model.Imp_Thai = Set()
model.Slack = Set()
##model.Biomass_st = Set()
##model.Gas_cc = Set()
##model.Gas_st = Set()


###Allocate generators that will ensure minimum reserves
model.ResGenerators = model.Coal_st | model.Oil_ic | model.Oil_st


######=================================================########
######               Segment B.2                       ########
######=================================================########

### Nodal sets
model.nodes = Set()
model.sources = Set(within=model.nodes)
model.sinks = Set(within=model.nodes)

model.h_nodes = Set()

model.d_nodes = Set()

model.gd_nodes = Set()
model.gn_nodes = Set()
model.td_nodes = Set()
model.tn_nodes = Set()



######=================================================########
######               Segment B.3                       ########
######=================================================########

#####==== Parameters for dispatchable resources ===####

#Generator type
model.typ = Param(model.Generators,within=Any)

#Node name
model.node = Param(model.Generators,within=Any)

#Max capacity
model.maxcap = Param(model.Generators,within=Any)

#Min capacity
model.mincap = Param(model.Generators,within=Any)

#Heat rate
model.heat_rate = Param(model.Generators,within=Any)

#Variable O&M
model.var_om = Param(model.Generators,within=Any)

#Fixed O&M cost
model.fix_om  = Param(model.Generators,within=Any)

#Start cost
model.st_cost = Param(model.Generators,within=Any)

#Ramp rate
model.ramp  = Param(model.Generators,within=Any)

#Minimun up time
model.minup = Param(model.Generators,within=Any)

#Minmun down time
model.mindn = Param(model.Generators,within=Any)

#Derate_factor as percent of maximum capacity of water-dependant generators
model.deratef = Param(model.Generators,within=NonNegativeReals)

#heat rates and import unit costs
model.gen_cost = Param(model.Generators,within=NonNegativeReals)


######=================================================########
######               Segment B.4                       ########
######=================================================########

######==== Transmission line parameters =======#######
model.linemva = Param(model.sources, model.sinks)
model.linesus = Param(model.sources, model.sinks)

### Transmission Loss as a %discount on production
model.TransLoss = Param(within=NonNegativeReals)

### Maximum line-usage as a percent of line-capacity
model.n1criterion = Param(within=NonNegativeReals)

### Minimum spinning reserve as a percent of total reserve
model.spin_margin = Param(within=NonNegativeReals)

model.m = Param(initialize = 1e5)

######=================================================########
######               Segment B.5                       ########
######=================================================########

######===== Parameters/initial_conditions to run simulation ======####### 
## Full range of time series information
model.SimHours = Param(within=PositiveIntegers)
model.SH_periods = RangeSet(1,model.SimHours+1)
model.SimDays = Param(within=PositiveIntegers)
model.SD_periods = RangeSet(1,model.SimDays+1)

# Operating horizon information 
model.HorizonHours = Param(within=PositiveIntegers)
model.HH_periods = RangeSet(0,model.HorizonHours)
model.hh_periods = RangeSet(1,model.HorizonHours)
model.ramp_periods = RangeSet(2,24)

######=================================================########
######               Segment B.6                       ########
######=================================================########

#Demand over simulation period
model.SimDemand = Param(model.d_nodes*model.SH_periods, within=NonNegativeReals)
#Horizon demand
model.HorizonDemand = Param(model.d_nodes*model.hh_periods,within=NonNegativeReals,mutable=True)

#Reserve for the entire system
model.SimReserves = Param(model.SH_periods, within=NonNegativeReals)
model.HorizonReserves = Param(model.hh_periods, within=NonNegativeReals,mutable=True)

##Variable resources over simulation period
model.SimHydro = Param(model.h_nodes, model.SH_periods, within=NonNegativeReals)
##model.SimSolar = Param(model.s_nodes, model.SH_periods, within=NonNegativeReals)
##model.SimWind = Param(model.w_nodes, model.SH_periods, within=NonNegativeReals)

#Variable resources over horizon
model.HorizonHydro = Param(model.h_nodes,model.hh_periods,within=NonNegativeReals,mutable=True)
##model.HorizonSolar = Param(model.s_nodes,model.hh_periods,within=NonNegativeReals,mutable=True)
##model.HorizonWind = Param(model.w_nodes,model.hh_periods,within=NonNegativeReals,mutable=True)

##Initial conditions
model.ini_on = Param(model.Generators, within=NonNegativeReals, mutable=True)


######=================================================########
######               Segment B.7                       ########
######=================================================########

######=======================Decision variables======================########
##Amount of day-ahead energy generated by each generator at each hour
model.mwh = Var(model.Generators,model.HH_periods,within=NonNegativeReals)

#1 if unit is on in hour i, otherwise 0
# def on_ini(model,j,i):
#     return(model.ini_on[j])
# model.on = Var(model.Generators,model.HH_periods, within=Binary, initialize=on_ini)
model.on = Var(model.Generators,model.HH_periods, within=Binary)

#1 if unit is switching on in hour i, otherwise 0
model.switch = Var(model.Generators,model.HH_periods, within=Binary)

#Amount of spining reserve offered by an unit in each hour
model.srsv = Var(model.Generators,model.HH_periods, within=NonNegativeReals)

#Amount of non-sping reserve offered by an unit in each hour
model.nrsv = Var(model.Generators,model.HH_periods, within=NonNegativeReals)

#dispatch of hydropower from each domestic dam in each hour
model.hydro = Var(model.h_nodes,model.HH_periods,within=NonNegativeReals)

###dispatch of solar-power in each hour
##model.solar = Var(model.s_nodes,model.HH_periods,within=NonNegativeReals)
##
###dispatch of wind-power in each hour
##model.wind = Var(model.w_nodes,model.HH_periods,within=NonNegativeReals)

#Voltage angle at each node in each hour
model.vlt_angle = Var(model.nodes,model.HH_periods)



######=================================================########
######               Segment B.8                       ########
######=================================================########

######================Objective function=============########

def SysCost(model):
    fixed = sum(model.maxcap[j]*model.fix_om[j]*model.on[j,i] for i in model.hh_periods for j in model.Generators)
    starts = sum(model.maxcap[j]*model.st_cost[j]*model.switch[j,i] for i in model.hh_periods for j in model.Generators)

    coal_st = sum(model.mwh[j,i]*(model.heat_rate[j]*model.gen_cost[j] + model.var_om[j]) for i in model.hh_periods for j in model.Coal_st)  
    oil_ic = sum(model.mwh[j,i]*(model.heat_rate[j]*model.gen_cost[j] + model.var_om[j]) for i in model.hh_periods for j in model.Oil_ic)
    oil_st = sum(model.mwh[j,i]*(model.heat_rate[j]*model.gen_cost[j] + model.var_om[j]) for i in model.hh_periods for j in model.Oil_st)

    imprt_v = sum(model.mwh[j,i]*model.gen_cost[j] for i in model.hh_periods for j in model.Imp_Viet)
    imprt_t = sum(model.mwh[j,i]*model.gen_cost[j] for i in model.hh_periods for j in model.Imp_Thai)

##    biomass_st = sum(model.mwh[j,i]*(model.heat_rate[j]*model.gen_cost[j] + model.var_om[j]) for i in model.hh_periods for j in model.Biomass_st)
##    gas_cc = sum(model.mwh[j,i]*(model.heat_rate[j]*model.gen_cost[j] + model.var_om[j]) for i in model.hh_periods for j in model.Gas_cc)
##    gas_st = sum(model.mwh[j,i]*(model.heat_rate[j]*model.gen_cost[j] + model.var_om[j]) for i in model.hh_periods for j in model.Gas_st)
    
    slack = sum(model.mwh[j,i]*model.heat_rate[j]*model.gen_cost[j] for i in model.hh_periods for j in model.Slack)
    
    return fixed +starts +coal_st +oil_ic +oil_st +imprt_v +imprt_t +slack  ## +biomass_st +gas_cc +gas_st

model.SystemCost = Objective(rule=SysCost, sense=minimize)



######=================================================########
######               Segment B.9                      ########
######=================================================########

######========== Logical Constraint =========#############
def OnCon(model,j,i):
    return model.mwh[j,i] <= model.on[j,i] * model.m
model.OnConstraint = Constraint(model.Generators,model.HH_periods,rule = OnCon)

def OnCon_initial(model,j,i):
    if i == 0:
        return (model.on[j,i] == model.ini_on[j])
    else:
      return Constraint.Skip
model.initial_value_constr = Constraint(model.Generators,model.HH_periods, rule=OnCon_initial)

def SwitchCon2(model,j,i):
    return model.switch[j,i] <= model.on[j,i] * model.m
model.Switch2Constraint = Constraint(model.Generators,model.hh_periods,rule = SwitchCon2)

def SwitchCon3(model,j,i):
    return  model.switch[j,i] <= (1 - model.on[j,i-1]) * model.m  
model.Switch3Constraint = Constraint(model.Generators,model.hh_periods,rule = SwitchCon3)

def SwitchCon4(model,j,i):
    return  model.on[j,i] - model.on[j,i-1] <= model.switch[j,i]
model.Switch4Constraint = Constraint(model.Generators,model.hh_periods,rule = SwitchCon4)


######========== Up/Down Time Constraint =========#############
##Min Up time
def MinUp(model,j,i,k):
    if i > 0 and k > i and k < min(i+model.minup[j]-1,model.HorizonHours):
        return model.on[j,i] - model.on[j,i-1] <= model.on[j,k]
    else: 
        return Constraint.Skip
model.MinimumUp = Constraint(model.Generators,model.HH_periods,model.HH_periods,rule=MinUp)

##Min Down time
def MinDown(model,j,i,k):
   if i > 0 and k > i and k < min(i+model.mindn[j]-1,model.HorizonHours):
       return model.on[j,i-1] - model.on[j,i] <= 1 - model.on[j,k]
   else:
       return Constraint.Skip
model.MinimumDown = Constraint(model.Generators,model.HH_periods,model.HH_periods,rule=MinDown)


######==========Ramp Rate Constraints =========#############
def Ramp1(model,j,i):
    a = model.mwh[j,i]
    b = model.mwh[j,i-1]
    return a - b <= model.ramp[j] 
model.RampCon1 = Constraint(model.Generators,model.ramp_periods,rule=Ramp1)

def Ramp2(model,j,i):
    a = model.mwh[j,i]
    b = model.mwh[j,i-1]
    return b - a <= model.ramp[j] 
model.RampCon2 = Constraint(model.Generators,model.ramp_periods,rule=Ramp2)


######=================================================########
######               Segment B.10                      ########
######=================================================########

######=========== Capacity Constraints ============##########
#Constraints for Max & Min Capacity of dispatchable resources
#derate factor can be below 1 for dry years, otherwise 1
def MaxC(model,j,i):
    return model.mwh[j,i]  <= model.on[j,i] * model.maxcap[j] *model.deratef[j]
model.MaxCap= Constraint(model.Generators,model.hh_periods,rule=MaxC)

def MinC(model,j,i):
    return model.mwh[j,i] >= model.on[j,i] * model.mincap[j]
model.MinCap= Constraint(model.Generators,model.hh_periods,rule=MinC)

#Max capacity constraints on domestic hydropower 
def HydroC(model,z,i):
    return model.hydro[z,i] <= model.HorizonHydro[z,i]  
model.HydroConstraint= Constraint(model.h_nodes,model.hh_periods,rule=HydroC)


######=================================================########
######               Segment B.11.1                    ########
######=================================================########

#########======================== Power balance in sub-station nodes (with/without demand) ====================#######
###With demand
def TDnodes_Balance(model,z,i):
    demand = model.HorizonDemand[z,i]
    impedance = sum(model.linesus[z,k] * (model.vlt_angle[z,i] - model.vlt_angle[k,i]) for k in model.sinks)   
    return - demand == impedance
model.TDnodes_BalConstraint= Constraint(model.td_nodes,model.hh_periods,rule= TDnodes_Balance)

###Without demand
def TNnodes_Balance(model,z,i):
    #demand = model.HorizonDemand[z,i]
    impedance = sum(model.linesus[z,k] * (model.vlt_angle[z,i] - model.vlt_angle[k,i]) for k in model.sinks)   
    return 0 == impedance
model.TNnodes_BalConstraint= Constraint(model.tn_nodes,model.hh_periods,rule= TNnodes_Balance)



######=================================================########
######               Segment B.11.2                    ########
######=================================================########

######=================== Power balance in nodes of variable resources (without demand in this case) =================########

###Hydropower Plants
def HPnodes_Balance(model,z,i):
    dis_hydro = model.hydro[z,i]
    #demand = model.HorizonDemand[z,i]
    impedance = sum(model.linesus[z,k] * (model.vlt_angle[z,i] - model.vlt_angle[k,i]) for k in model.sinks)
    return (1 - model.TransLoss) * dis_hydro == impedance ##- demand
model.HPnodes_BalConstraint= Constraint(model.h_nodes,model.hh_periods,rule= HPnodes_Balance)


######=================================================########
######               Segment B.11.3                    ########
######=================================================########

##########============ Power balance in nodes of dispatchable resources with demand ==============############
def GD1_Balance(model,i):
    gd = 1
    thermo = sum(model.mwh[j,i] for j in model.GD1Gens)
    demand = model.HorizonDemand[gd_nodes[gd-1],i]
    impedance = sum(model.linesus[gd_nodes[gd-1],k] * (model.vlt_angle[gd_nodes[gd-1],i] - model.vlt_angle[k,i]) for k in model.sinks)   
    return (1 - model.TransLoss) * thermo - demand == impedance
model.GD1_BalConstraint= Constraint(model.hh_periods,rule= GD1_Balance)

def GD2_Balance(model,i):
    gd = 2
    thermo = sum(model.mwh[j,i] for j in model.GD2Gens)
    demand = model.HorizonDemand[gd_nodes[gd-1],i]
    impedance = sum(model.linesus[gd_nodes[gd-1],k] * (model.vlt_angle[gd_nodes[gd-1],i] - model.vlt_angle[k,i]) for k in model.sinks)   
    return (1 - model.TransLoss) * thermo - demand == impedance
model.GD2_BalConstraint= Constraint(model.hh_periods,rule= GD2_Balance)

def GD3_Balance(model,i):
    gd = 3
    thermo = sum(model.mwh[j,i] for j in model.GD3Gens)
    demand = model.HorizonDemand[gd_nodes[gd-1],i]
    impedance = sum(model.linesus[gd_nodes[gd-1],k] * (model.vlt_angle[gd_nodes[gd-1],i] - model.vlt_angle[k,i]) for k in model.sinks)   
    return (1 - model.TransLoss) * thermo - demand == impedance
model.GD3_BalConstraint= Constraint(model.hh_periods,rule= GD3_Balance)

def GD4_Balance(model,i):
    gd = 4
    thermo = sum(model.mwh[j,i] for j in model.GD4Gens)
    demand = model.HorizonDemand[gd_nodes[gd-1],i]
    impedance = sum(model.linesus[gd_nodes[gd-1],k] * (model.vlt_angle[gd_nodes[gd-1],i] - model.vlt_angle[k,i]) for k in model.sinks)   
    return (1 - model.TransLoss) * thermo - demand == impedance
model.GD4_BalConstraint= Constraint(model.hh_periods,rule= GD4_Balance)

def GD5_Balance(model,i):
    gd = 5
    thermo = sum(model.mwh[j,i] for j in model.GD5Gens)
    demand = model.HorizonDemand[gd_nodes[gd-1],i]
    impedance = sum(model.linesus[gd_nodes[gd-1],k] * (model.vlt_angle[gd_nodes[gd-1],i] - model.vlt_angle[k,i]) for k in model.sinks)   
    return (1 - model.TransLoss) * thermo - demand == impedance
model.GD5_BalConstraint= Constraint(model.hh_periods,rule= GD5_Balance)

def GD6_Balance(model,i):
    gd = 6
    thermo = sum(model.mwh[j,i] for j in model.GD6Gens)
    demand = model.HorizonDemand[gd_nodes[gd-1],i]
    impedance = sum(model.linesus[gd_nodes[gd-1],k] * (model.vlt_angle[gd_nodes[gd-1],i] - model.vlt_angle[k,i]) for k in model.sinks)   
    return (1 - model.TransLoss) * thermo - demand == impedance
model.GD6_BalConstraint= Constraint(model.hh_periods,rule= GD6_Balance)

def GD7_Balance(model,i):
    gd = 7
    thermo = sum(model.mwh[j,i] for j in model.GD7Gens)
    demand = model.HorizonDemand[gd_nodes[gd-1],i]
    impedance = sum(model.linesus[gd_nodes[gd-1],k] * (model.vlt_angle[gd_nodes[gd-1],i] - model.vlt_angle[k,i]) for k in model.sinks)   
    return (1 - model.TransLoss) * thermo - demand == impedance
model.GD7_BalConstraint= Constraint(model.hh_periods,rule= GD7_Balance)

def GD8_Balance(model,i):
    gd = 8
    thermo = sum(model.mwh[j,i] for j in model.GD8Gens)
    demand = model.HorizonDemand[gd_nodes[gd-1],i]
    impedance = sum(model.linesus[gd_nodes[gd-1],k] * (model.vlt_angle[gd_nodes[gd-1],i] - model.vlt_angle[k,i]) for k in model.sinks)   
    return (1 - model.TransLoss) * thermo - demand == impedance
model.GD8_BalConstraint= Constraint(model.hh_periods,rule= GD8_Balance)

def GD9_Balance(model,i):
    gd = 9
    thermo = sum(model.mwh[j,i] for j in model.GD9Gens)
    demand = model.HorizonDemand[gd_nodes[gd-1],i]
    impedance = sum(model.linesus[gd_nodes[gd-1],k] * (model.vlt_angle[gd_nodes[gd-1],i] - model.vlt_angle[k,i]) for k in model.sinks)   
    return (1 - model.TransLoss) * thermo - demand == impedance
model.GD9_BalConstraint= Constraint(model.hh_periods,rule= GD9_Balance)



##########============ Power balance in nodes of dispatchable resources without demand ==============############
def GN1_Balance(model,i):
    gn = 1
    thermo = sum(model.mwh[j,i] for j in model.GN1Gens)    
    impedance = sum(model.linesus[gn_nodes[gn-1],k] * (model.vlt_angle[gn_nodes[gn-1],i] - model.vlt_angle[k,i]) for k in model.sinks)   
    return (1 - model.TransLoss) * thermo == impedance #- demand
model.GN1_BalConstraint= Constraint(model.hh_periods,rule= GN1_Balance)

def GN2_Balance(model,i):
    gn = 2
    thermo = sum(model.mwh[j,i] for j in model.GN2Gens)    
    impedance = sum(model.linesus[gn_nodes[gn-1],k] * (model.vlt_angle[gn_nodes[gn-1],i] - model.vlt_angle[k,i]) for k in model.sinks)   
    return (1 - model.TransLoss) * thermo == impedance #- demand
model.GN2_BalConstraint= Constraint(model.hh_periods,rule= GN2_Balance)

def GN3_Balance(model,i):
    gn = 3
    thermo = sum(model.mwh[j,i] for j in model.GN3Gens)    
    impedance = sum(model.linesus[gn_nodes[gn-1],k] * (model.vlt_angle[gn_nodes[gn-1],i] - model.vlt_angle[k,i]) for k in model.sinks)   
    return (1 - model.TransLoss) * thermo == impedance #- demand
model.GN3_BalConstraint= Constraint(model.hh_periods,rule= GN3_Balance)



######=================================================########
######               Segment B.12                    ########
######=================================================########

######==================Transmission  constraints==================########

####=== Reference Node =====#####
def ref_node(model,i):
    return model.vlt_angle['GS1',i] == 0
model.Ref_NodeConstraint= Constraint(model.hh_periods,rule= ref_node)


######========== Transmission Capacity Constraints (N-1 Criterion) =========#############
def MaxLine(model,s,k,i):
    if model.linemva[s,k] > 0:
        return (model.n1criterion) * model.linemva[s,k] >= model.linesus[s,k] * (model.vlt_angle[s,i] - model.vlt_angle[k,i])
    else:
        return Constraint.Skip
model.MaxLineConstraint= Constraint(model.sources,model.sinks,model.hh_periods,rule=MaxLine)

def MinLine(model,s,k,i):
    if model.linemva[s,k] > 0:
        return (-model.n1criterion) * model.linemva[s,k] <= model.linesus[s,k] * (model.vlt_angle[s,i] - model.vlt_angle[k,i])
    else:
        return Constraint.Skip
model.MinLineConstraint= Constraint(model.sources,model.sinks,model.hh_periods,rule=MinLine)



######=================================================########
######               Segment B.13                      ########
######=================================================########

######===================Reserve and zero-sum constraints ==================########

##System Reserve Requirement
def SysReserve(model,i):
    return sum(model.srsv[j,i] for j in model.ResGenerators) + sum(model.nrsv[j,i] for j in model.ResGenerators) >= model.HorizonReserves[i]
model.SystemReserve = Constraint(model.hh_periods,rule=SysReserve)

##Spinning Reserve Requirement
def SpinningReq(model,i):
    return sum(model.srsv[j,i] for j in model.ResGenerators) >= model.spin_margin * model.HorizonReserves[i] 
model.SpinReq = Constraint(model.hh_periods,rule=SpinningReq)           

##Spinning reserve can only be offered by units that are online
def SpinningReq2(model,j,i):
    return model.srsv[j,i] <= model.on[j,i]*model.maxcap[j] *model.deratef[j]
model.SpinReq2= Constraint(model.Generators,model.hh_periods,rule=SpinningReq2) 

##Non-Spinning reserve can only be offered by units that are offline
def NonSpinningReq(model,j,i):
    return model.nrsv[j,i] <= (1 - model.on[j,i])*model.maxcap[j] *model.deratef[j]
model.NonSpinReq= Constraint(model.Generators,model.hh_periods,rule=NonSpinningReq)


######========== Zero Sum Constraint =========#############
def ZeroSum(model,j,i):
    return model.mwh[j,i] + model.srsv[j,i] + model.nrsv[j,i] <= model.maxcap[j]
model.ZeroSumConstraint=Constraint(model.Generators,model.hh_periods,rule=ZeroSum)


######======================================#############
######==========        End        =========#############
######=======================================############

