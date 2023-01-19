import numpy as np
import os
import scipy.io as scio
from scipy.optimize import minimize
import importlib
#from ax import optimize as axoptimize
#import nevergrad as ng


def descente(f_cout,pas,start,npas,*args):

    # Descente de gradient bien bien pourrie.
    # A éviter au maximum, elle est vraiment nulle.
    # Gestion des arguments optionnels et des optionnalité des arguments
    limites=False
    #print(len(args))
    if (len(args)==2):
        xmin=args[0]
        xmax=args[1]
        limites=True
#        print("Limites !")
    n=start.size
    if (len(pas)!=n):
        pas=pas*np.ones(n)

    # Initialisation
    convergence=np.zeros(npas)
    x=start
    f=f_cout(x)
    direction=np.zeros(n)

    for k in range(0,npas):
        dpas=np.zeros(n)
        for j in range(0,n):
            dpas[j]=pas[j]/100
            direction[j]=(f-f_cout(x+dpas))/dpas[j]
            dpas[j]=0
        # Determination du nouveau x - prenant les limites en compte.
        xn=x+pas*direction
        if (limites):
            super=(xn<xmax)
            infra=(xn>xmin)
            xn=xn*super*infra+xmin*(1-infra)+xmax*(1-super)
        # Calcul de sa fonction de coût
        tmp=f_cout(xn)
        # Est-ce qu'on avance ou est-ce qu'on divise le pas par deux ?
        if (tmp<f):
            f=tmp
            x=xn
        else:
            #print('Division du pas')
            pas=pas/2
            # Si le pas est trop petit, on est sans doute sur le minimum.
            # On sort.
            if (max(pas)<1e-10):
                convergence=convergence[0:k]
                break
        convergence[k]=f
    return [x,convergence,x]

convergence = []
vals = []
bestx= None
bestv = float("inf")

def descente2(f_cout,pas,start,npas,*args):
    if (len(args)==2):
        xmin=args[0]
        xmax=args[1]
        limites=True
    n = start.size
    global convergence
    x = np.array(start)
    epsilon = 1.e-7
    convergence = []
    def jac(x):
        grad = np.zeros(n)
        global convergence
        val = f_cout(x)
        convergence += [val]
        for i in range(n):
            xp = np.array(x)
            xp[i] = xp[i] + epsilon
            grad[i] = (f_cout(xp) - val) / epsilon
        return grad
    global vals
    vals = []
    def f_cout2(x):
        global vals
        global bestx
        global bestv
        v = f_cout(x)
        vals += [v]
        if v < bestv and len(vals) <= npas:
            bestx= np.array(x)
            bestv=v
        return vals[-1]
        
    res = minimize(f_cout2, start, method='L-BFGS-B', jac=jac, tol=1e-99,
             options={'disp': False, 'maxiter': npas}, bounds=[(xmin[i], xmax[i]) for i in range(len(xmin))])
    x = res.x
    #assert len(convergence) == npas, f"{len(convergence)} == {npas} {len(vals)}"
    return [bestx,[min(convergence[:(i+1)]) for i in range(npas)], bestx]


def ngopt(f_cout,budget,X_min,X_max,population,algorithm):
#    import sys
#    try:
#       sys.modules.pop('nevergrad')
#       sys.modules.pop('nevergrad')
#       sys.modules.pop('nevergrad')
#    except:
#       pass   # this is so dirty
#    try:
#       sys.path.remove("/private/home/oteytaud/ant/nn")
#       sys.path.remove("/private/home/oteytaud/ant/nn")
#       sys.path.remove("/private/home/oteytaud/ant/nn")
#    except:
#       pass   # this is so dirty
#    try:
#       sys.path.remove("/private/home/oteytaud/ant/on")
#       sys.path.remove("/private/home/oteytaud/ant/on")
#       sys.path.remove("/private/home/oteytaud/ant/on")
#    except:
#       pass   # this is so dirty
#    try:
#       #os.system('touch nevergrad')
#       #os.system('mv nevergrad from_ages_nevergrad'+str(np.random.randint(5000)))
#       #os.system('ln -s newnevergrad nevergrad')
#       sys.path.append("/private/home/oteytaud/ant/nn")
#       import nevergrad as ng
#       #print(f"sys.path= {sys.path}, reg={list(ng.optimizers.registry.keys())}")
#       ngopt = ng.optimizers.registry[algorithm](ng.p.Array(shape=(len(X_min),), lower=X_min, upper=X_max), budget=budget)
#    except:
#       sys.modules.pop('nevergrad')
#       try:
#          sys.path.remove("/private/home/oteytaud/ant/nn")
#       except:
#          pass   # this is so dirty
#       #os.system('touch nevergrad')
#       #os.system('mv nevergrad from_ages_nevergrad'+str(np.random.randint(5000)))
#       #os.system('ln -s oldnevergrad nevergrad')
#       sys.modules.pop('nevergrad')
#       sys.path.append("/private/home/oteytaud/ant/on")
#       import nevergrad as ng
#       #print(f"sys.path= {sys.path}, reg={list(ng.optimizers.registry.keys())}")
#
    global vals
    global bestv
    global bestx
    bestv = float("inf")
    vals = []
    def f_cout2(x):
        global vals
        global bestx
        global bestv
        v = f_cout(x)
        vals += [v]
        if v < bestv and len(vals) <= budget:
            bestx= np.array(x)
            bestv=v
        return vals[-1]
    if algorithm in ng.optimizers.registry:
      ngopt = ng.optimizers.registry[algorithm](ng.p.Array(shape=(len(X_min),), lower=X_min, upper=X_max), budget=budget)
      try:
           ngopt.minimize(f_cout2)
           recom = ngopt.recommend().value
      except:
           print(f"pb in {algorithm}")
           recom = 60. *np.ones(len(X_min))
      #print(f"We{algorithm} recommend {recom}")
    else:
      if algorithm == "AX":
         def ax_obj(p):
            data = [p["x" + str(i)] for i in range(len(X_min))]
            return float(f_cout2(data))
         parameters = [{"name": "x"+str(i), "type":"range", "bounds":[float(X_min[i]), float(X_max[i])]} for i in range(len(X_min))]
         best_parameters, best_values, experiment, model = axoptimize(
                                             parameters,
                                             evaluation_function = ax_obj,
                                             minimize=True,
                                             total_trials = budget)
         recom = np.array([best_parameters["x" + str(i)] for i in range(len(X_min))])
         #print(f"AX:We recommend {recom}")
      else:
         assert False, f"{algorithm} is missing"

    #print(f"End: we recommend {recom}")
    return [bestx, vals, recom]


def DEvol(f_cout,budget,X_min,X_max,population):

# Ce DE est un current to best
# Hypertuné sur le problème chirped
# Elimination brutale des individus ne respectant pas les bornes
# (on pourrait comparer à ce qui se passe si on remet juste au bord
# ça pourrait être une bonne idée sur certains problèmes)

# Paramètres de DE - paramètres potentiels de la fonction
    cr=0.5; # Chances de passer les paramètres du parent à son rejeton.
    f1=0.9;
    f2=0.8;
    n=X_min.size

    # Initialisation de la population
    omega=np.zeros((population,n))
    cost=np.zeros(population)
    # Tirage aléatoire dans le domaine défini par X_min et X_max.
    for k in range(0,population):
        omega[k]=X_min+(X_max-X_min)*np.random.random_sample(n)
        cost[k]=f_cout(omega[k])

    # Who's the best ?
    who=np.argmin(cost)
    best=omega[who]
    # Initialisations
    evaluation=population
    convergence=[]
    generation=0
    convergence.append(cost[who])

    # Boucle DE
    while evaluation<budget-population:
        for k in range(0,population):
            crossover=(np.random.random_sample(n)<cr)
            X=(omega[k]+f1*(omega[np.random.randint(population)]-omega[np.random.randint(population)])+f2*(best-omega[k]))*(1-crossover)+crossover*omega[k]
            if np.prod((X>X_min)*(X<X_max)):
                tmp=f_cout(X)
                evaluation=evaluation+1
                if (tmp<cost[k]) :
                    cost[k]=tmp
                    omega[k]=X

        generation=generation+1
        #print('generation:',generation,'evaluations:',evaluation)
        #
        who=np.argmin(cost)
        best=omega[who]
        convergence.append(cost[who])

    convergence=convergence[0:generation+1]

    return [best,convergence,best]


def DEvol_bord(f_cout,budget,X_min,X_max,population):

# Ce DE est un current to best
# Hypertuné sur le problème chirped
# Elimination brutale des individus ne respectant pas les bornes
# Avec remise aux bords !!
# Paramètres de DE - paramètres potentiels de la fonction
    cr=0.5; # Chances de passer les paramètres du parent à son rejeton.
    f1=0.9;
    f2=0.8;
    n=X_min.size

    # Initialisation de la population
    omega=np.zeros((population,n))
    cost=np.zeros(population)
    # Tirage aléatoire dans le domaine défini par X_min et X_max.
    for k in range(0,population):
        omega[k]=X_min+(X_max-X_min)*np.random.random_sample(n)
        cost[k]=f_cout(omega[k])

    # Who's the best ?
    who=np.argmin(cost)
    best=omega[who]
    # Initialisations
    evaluation=population
    convergence=[]
    generation=0
    convergence.append(cost[who])

    # Boucle DE
    while evaluation<budget-population:
        for k in range(0,population):
            crossover=(np.random.random_sample(n)<cr)
            X=(omega[k]+f1*(omega[np.random.randint(population)]-omega[np.random.randint(population)])+f2*(best-omega[k]))*(1-crossover)+crossover*omega[k]
            # Remise aux bords, c'est là que ça se fait !
            X = np.maximum(X,X_min)
            X = np.minimum(X,X_max)
            tmp=f_cout(X)
            evaluation=evaluation+1
            if (tmp<cost[k]) :
                cost[k]=tmp
                omega[k]=X
        generation=generation+1
        #print('generation:',generation,'evaluations:',evaluation)
        #
        who=np.argmin(cost)
        best=omega[who]
        convergence.append(cost[who])

    convergence=convergence[0:generation+1]

    return [best,convergence,best]

def DEvol_struct_bragg(f_cout,budget,X_min,X_max,population):

# Ce DE est un current to best
# Hypertuné sur le problème chirped
# Elimination brutale des individus ne respectant pas les bornes
# (on pourrait comparer à ce qui se passe si on remet juste au bord
# ça pourrait être une bonne idée sur certains problèmes)

# Paramètres de DE - paramètres potentiels de la fonction
    cr=0.5; # Chances de passer les paramètres du parent à son rejeton.
    f1=0.9;
    f2=0.8;
    n=X_min.size

    # Initialisation de la population
    omega=np.zeros((population,n))
    cost=np.zeros(population)
    # Tirage aléatoire dans le domaine défini par X_min et X_max.
    for k in range(0,population):
        omega[k]=X_min+(X_max-X_min)*np.random.random_sample(n)
        cost[k]=f_cout(omega[k])

    # Who's the best ?
    who=np.argmin(cost)
    best=omega[who]
    # Initialisations
    evaluation=population
    convergence=[]
    generation=0
    convergence.append(cost[who])

    # Boucle DE
    while evaluation<budget-population:
        for k in range(0,population):
            # For Bragg: the crossover is duplicated so that the permittivity
            # and the thickness of a given layer are transmitted or not
            # but never separated
            tmp = (np.random.random_sample(n//2)<cr)
            crossover= np.concatenate((tmp,tmp))
            X=(omega[k]+f1*(omega[np.random.randint(population)]-omega[np.random.randint(population)])+f2*(best-omega[k]))*(1-crossover)+crossover*omega[k]
            if np.prod((X>X_min)*(X<X_max)):
                tmp=f_cout(X)
                evaluation=evaluation+1
                if (tmp<cost[k]) :
                    cost[k]=tmp
                    omega[k]=X

        generation=generation+1
        #print('generation:',generation,'evaluations:',evaluation)
        #
        who=np.argmin(cost)
        best=omega[who]
        convergence.append(cost[who])

    convergence=convergence[0:generation+1]

    return [best,convergence,best]

def DEvol_struct_morpho(f_cout,budget,X_min,X_max,population):

# Ce DE est un current to best
# Hypertuné sur le problème chirped
# Elimination brutale des individus ne respectant pas les bornes
# (on pourrait comparer à ce qui se passe si on remet juste au bord
# ça pourrait être une bonne idée sur certains problèmes)

# Paramètres de DE - paramètres potentiels de la fonction
    cr=0.5; # Chances de passer les paramètres du parent à son rejeton.
    f1=0.9;
    f2=0.8;
    n=X_min.size

    # Initialisation de la population
    omega=np.zeros((population,n))
    cost=np.zeros(population)
    # Tirage aléatoire dans le domaine défini par X_min et X_max.
    for k in range(0,population):
        omega[k]=X_min+(X_max-X_min)*np.random.random_sample(n)
        cost[k]=f_cout(omega[k])

    # Who's the best ?
    who=np.argmin(cost)
    best=omega[who]
    # Initialisations
    evaluation=population
    convergence=[]
    generation=0
    convergence.append(cost[who])

    # Boucle DE
    while evaluation<budget-population:
        for k in range(0,population):
            # For Bragg: the crossover is duplicated so that the permittivity
            # and the thickness of a given layer are transmitted or not
            # but never separated
            tmp = (np.random.random_sample(n//4)<cr)
            crossover= np.concatenate((tmp,tmp,tmp,tmp))
            X=(omega[k]+f1*(omega[np.random.randint(population)]-omega[np.random.randint(population)])+f2*(best-omega[k]))*(1-crossover)+crossover*omega[k]
            if np.prod((X>X_min)*(X<X_max)):
                tmp=f_cout(X)
                evaluation=evaluation+1
                if (tmp<cost[k]) :
                    cost[k]=tmp
                    omega[k]=X

        generation=generation+1
        #print('generation:',generation,'evaluations:',evaluation)
        #
        who=np.argmin(cost)
        best=omega[who]
        convergence.append(cost[who])

    convergence=convergence[0:generation+1]

    return [best,convergence,best]
