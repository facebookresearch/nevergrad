import numpy as np
from vdc import vdc

broad = 250.
counter = 0
budget = -1
default_rando = None
default_num = None
current_best = None
current_best_val = float("inf")

def bragg(X):
    # La première moitié du vecteur X concerne les permittivités (epsilon)
    # des différentes coucches.
    # La deuxième concerne les épaisseurs (en nm) des différentes couches
    # Attention, au delà de 150 couches, ça peut devenir moins précis
    # numériquement
    # Remarques :
    # On peut faire de la structuration avec Bragg.
    # On peut regarder l'influence de la remise au bords pour DE.
    #print(X)
    lam=600
#    assert False
    bar=int(np.size(X)/2)
    n=np.concatenate(([1],np.sqrt(X[0:bar]),[1.7320508075688772]))
    Type=np.arange(0,bar+2)
    hauteur=np.concatenate(([0],X[bar:2*bar],[0]))
    tmp=np.tan(2*np.pi*n[Type]*hauteur/lam)
    #Specific to this substrate.
    Z=n[-1]
    for k in range(np.size(Type)-1,0,-1):
        Z=(Z-1j*n[Type[k]]*tmp[k])/(1-1j*tmp[k]*Z/n[Type[k]])
    #Specific to air.
    r=(1-Z)/(1+Z)
    c=np.real(1-r*np.conj(r))
    return c

vdc_index = 0

def bragg_fixed(X, num=1, rando=False, check_count=True):
    global current_best_val
    global current_best
    global counter
    global budget
    global vdc_counter
    global broad
    if default_rando is not None:
        rando = default_rando
    if default_num is not None and num == 1:
        num = default_num
    #if num > 100:
    #    print(f"Disc={num}")
    #print(X)
    #lam=600
    if X is None:
        counter = 0
        return -1.
    global vdc_index
    if counter == 0:
       vdc_index = 0
    if num > budget - counter and check_count:
        num = budget - counter
    if num <= 0:
        return float("inf")
    epsilon = broad / num
    #lams = [400 + (i + .5) * epsilon for i in range(num)]
    if rando < 0.:
      r = (-rando * np.random.rand()) + (1+rando)*.5
    else:
      r = (rando * vdc(vdc_index)) + (1-rando)*.5
      vdc_index += 1
    lams = [400 + (i + r) * epsilon for i in range(num)]
    c = []
    for lam in lams:
        bar=len(X)
        counter += 1
        if counter > budget and check_count:
            return float("inf")
        n=np.array([1]+[1.7320508075688772,1.4142135623730951]*int(bar/2)+[1.7320508075688772])
        Type=np.arange(0,bar+2)
        hauteur=np.concatenate(([0],X,[0]))
        tmp=np.tan(2*np.pi*n[Type]*hauteur/lam)
        #Specific to this substrate.
        Z=n[-1]
        for k in range(np.size(Type)-1,0,-1):
            Z=(Z-1j*n[Type[k]]*tmp[k])/(1-1j*tmp[k]*Z/n[Type[k]])
        #Specific to air.
        r=(1-Z)/(1+Z)
        c+=[np.real(1-r*np.conj(r))]
    #if len(lams) > 1 and np.random.rand() < 0.001:
    #   print(f"lams={lams}, c={c}")
    #val = np.sum(c) / len(lams)
    val = np.max(c)
    if len(lams) > 1000:
       if val < current_best_val:
           current_best_val = val
           print(val, " obtained by ", list(X))

    return val

# Bragg with fixed refractive index.
def bragg_origin(X):
    #print(X)
    lam=600
    bar=len(X)
    n=np.array([1]+[1.7320508075688772,1.4142135623730951]*int(bar/2)+[1.7320508075688772])
    Type=np.arange(0,bar+2)
    hauteur=np.concatenate(([0],X,[0]))
    tmp=np.tan(2*np.pi*n[Type]*hauteur/lam)
    #Specific to this substrate.
    Z=n[-1]
    for k in range(np.size(Type)-1,0,-1):
        Z=(Z-1j*n[Type[k]]*tmp[k])/(1-1j*tmp[k]*Z/n[Type[k]])
    #Specific to air.
    r=(1-Z)/(1+Z)
    c=np.real(1-r*np.conj(r))
    return c

def chirped(X):
    lam=np.linspace(500,800,50)
    n=np.array([1,1.4142135623730951,1.7320508075688772])
    Type=np.concatenate(([0],np.tile([2,1],int(np.size(X)/2)),[2]))
    hauteur=np.concatenate(([0],X,[0]))
    r=np.zeros(np.size(lam))
    for m in range(0,np.size(lam)):
        #Specific to this substrate.
        tmp=np.tan(2*np.pi*n[Type]*hauteur/lam[m])
        Z=1.7320508075688772
        for k in range(np.size(Type)-1,0,-1):
            Z=(Z-1j*n[Type[k]]*tmp[k])/(1-1j*tmp[k]*Z/n[Type[k]])
        #Specific to air.
        r[m]=abs((1-Z)/(1+Z))**2
    c=1-np.mean(r)
    return c

def cascade(T,U):
    n=int(T.shape[1]/2)
    J=np.linalg.inv(np.eye(n)-np.matmul(U[0:n,0:n],T[n:2*n,n:2*n]))
    K=np.linalg.inv(np.eye(n)-np.matmul(T[n:2*n,n:2*n],U[0:n,0:n]))
    S=np.block([[T[0:n,0:n]+np.matmul(np.matmul(np.matmul(T[0:n,n:2*n],J),U[0:n,0:n]),T[n:2*n,0:n]),np.matmul(np.matmul(T[0:n,n:2*n],J),U[0:n,n:2*n])],[np.matmul(np.matmul(U[n:2*n,0:n],K),T[n:2*n,0:n]),U[n:2*n,n:2*n]+np.matmul(np.matmul(np.matmul(U[n:2*n,0:n],K),T[n:2*n,n:2*n]),U[0:n,n:2*n])]])
    return S

def c_bas(A,V,h):
    n=int(A.shape[1]/2)
    D=np.diag(np.exp(1j*V*h))
    S=np.block([[A[0:n,0:n],np.matmul(A[0:n,n:2*n],D)],[np.matmul(D,A[n:2*n,0:n]),np.matmul(np.matmul(D,A[n:2*n,n:2*n]),D)]])
    return S

def marche(a,b,p,n,x):
    from scipy.linalg import toeplitz
    l=np.zeros(n,dtype=np.complex)
    m=np.zeros(n,dtype=np.complex)
    tmp=1/(2*np.pi*np.arange(1,n))*(np.exp(-2*1j*np.pi*p*np.arange(1,n))-1)*np.exp(-2*1j*np.pi*np.arange(1,n)*x)
    l[1:n]=1j*(a-b)*tmp
    l[0]=p*a+(1-p)*b
    m[0]=l[0]
    m[1:n]=1j*(b-a)*np.conj(tmp)
    T=toeplitz(l,m)
    return T

def creneau(k0,a0,pol,e1,e2,a,n,x0):
    nmod=int(n/2)
    alpha=np.diag(a0+2*np.pi*np.arange(-nmod,nmod+1))
    if (pol==0):
        M=alpha*alpha-k0*k0*marche(e1,e2,a,n,x0)
        L,E=np.linalg.eig(M)
        L=np.sqrt(-L+0j)
        L=(1-2*(np.imag(L)<-1e-15))*L
        P=np.block([[E],[np.matmul(E,np.diag(L))]])
    else:
        U=marche(1/e1,1/e2,a,n,x0)
        T=np.linalg.inv(U)
        M=np.matmul(np.matmul(np.matmul(T,alpha),np.linalg.inv(marche(e1,e2,a,n,x0))),alpha)-k0*k0*T
        L,E=np.linalg.eig(M)
        L=np.sqrt(-L+0j)
        L=(1-2*(np.imag(L)<-1e-15))*L
        P=np.block([[E],[np.matmul(np.matmul(U,E),np.diag(L))]])
    return P,L

def homogene(k0,a0,pol,epsilon,n):
    nmod=int(n/2)
    valp=np.sqrt(epsilon*k0*k0-(a0+2*np.pi*np.arange(-nmod,nmod+1))**2+0j)
    valp=valp*(1-2*(valp<0))*(pol/epsilon+(1-pol))
    P=np.block([[np.eye(n)],[np.diag(valp)]])
    return P,valp

def interface(P,Q):
    n=int(P.shape[1])
    S=np.matmul(np.linalg.inv(np.block([[P[0:n,0:n],-Q[0:n,0:n]],[P[n:2*n,0:n],Q[n:2*n,0:n]]])),np.block([[-P[0:n,0:n],Q[0:n,0:n]],[P[n:2*n,0:n],Q[n:2*n,0:n]]]))
    return S

def morpho(X):
    lam=449.5897
    pol=1.
    d=600.521475
    nmod=25
    #nmod=1
    e2=2.4336
    n=2*nmod+1
    n_motifs=int(X.size/4)
    X=X/d
    h=X[0:n_motifs]
    x0=X[n_motifs:2*n_motifs]
    a=X[2*n_motifs:3*n_motifs]
    spacers=X[3*n_motifs:4*n_motifs]
    l=lam/d
    k0=2*np.pi/l
    P,V=homogene(k0,0,pol,1,n)
    S=np.block([[np.zeros([n,n]),np.eye(n,dtype=np.complex)],[np.eye(n),np.zeros([n,n])]])
    for j in range(0,n_motifs):
        Pc,Vc=creneau(k0,0,pol,e2,1,a[j],n,x0[j])
        S=cascade(S,interface(P,Pc))
        S=c_bas(S,Vc,h[j])
        S=cascade(S,interface(Pc,P))
        S=c_bas(S,V,spacers[j])
    Pc,Vc=homogene(k0,0,pol,e2,n)
    S=cascade(S,interface(P,Pc))
    R=np.zeros(3,dtype=np.float)
    for j in range(-1,2):
        R[j]=(abs(S[j+nmod,nmod])**2*np.real(V[j+nmod])/k0)
    cost=1-(R[-1]+R[1])/2+R[0]/2

    lam=(np.array([400,500,600,700,800])+0.24587)/d
    bar=0
    for lo in lam:
        k0=2*np.pi/lo
        P,V=homogene(k0,0,pol,1,n)
        S=np.block([[np.zeros([n,n],dtype=np.complex),np.eye(n)],[np.eye(n),np.zeros([n,n])]])
        for j in range(0,n_motifs):
            Pc,Vc=creneau(k0,0,pol,e2,1,a[j],n,x0[j])
            S=cascade(S,interface(P,Pc))
            S=c_bas(S,Vc,h[j])
            S=cascade(S,interface(Pc,P))
            S=c_bas(S,V,spacers[j])
        Pc,Vc=homogene(k0,0,pol,e2,n)
        S=cascade(S,interface(P,Pc))
        bar+=abs(S[nmod,nmod])**2*np.real(V[nmod])/k0
    cost+=bar/lam.size
    return cost

def neomorpho(X):
    # Nouvelle version de Morpho ::
    # Moins d'évaluation de la fonction de coût
    # Toujours juste le TM (discutable, ça)
    # Suffit normalement pour faire apparaître l'interdigitation

    lam=449.5897
    pol=0.
    d=600.521475
    nmod=25
    #nmod=1
    e2=2.4336
    n=2*nmod+1
    n_motifs=int(X.size/4)
    X=X/d
    h=X[0:n_motifs]
    x0=X[n_motifs:2*n_motifs]
    a=X[2*n_motifs:3*n_motifs]
    spacers=X[3*n_motifs:4*n_motifs]
    l=lam/d
    k0=2*np.pi/l
    P,V=homogene(k0,0,pol,1,n)
    S=np.block([[np.zeros([n,n]),np.eye(n,dtype=np.complex)],[np.eye(n),np.zeros([n,n])]])
    for j in range(0,n_motifs):
        Pc,Vc=creneau(k0,0,pol,e2,1,a[j],n,x0[j])
        S=cascade(S,interface(P,Pc))
        S=c_bas(S,Vc,h[j])
        S=cascade(S,interface(Pc,P))
        S=c_bas(S,V,spacers[j])
    Pc,Vc=homogene(k0,0,pol,e2,n)
    S=cascade(S,interface(P,Pc))
    R=np.zeros(3,dtype=np.float)
    for j in range(-1,2):
        R[j]=(abs(S[j+nmod,nmod])**2*np.real(V[j+nmod])/k0)
    cost=1-(R[-1]+R[1])/2+R[0]/2
    return cost
