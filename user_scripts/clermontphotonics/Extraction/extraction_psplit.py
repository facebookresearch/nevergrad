import numpy as np
import matplotlib.pyplot as plt

def cascade(T,U):
    '''Cascading of two scattering matrices T and U.
    Since T and U are scattering matrices, it is expected that they are square
    and have the same dimensions which are necessarily EVEN.
    '''

    n=int(T.shape[1] / 2)
    J=np.linalg.inv( np.eye(n) - np.matmul(U[0:n,0:n],T[n:2*n,n:2*n] ) )
    K=np.linalg.inv( np.eye(n) - np.matmul(T[n:2*n,n:2*n],U[0:n,0:n] ) )
    S=np.block([[T[0:n,0:n] + np.matmul(np.matmul(np.matmul(T[0:n,n:2*n],J),
    U[0:n,0:n]),T[n:2*n,0:n]),np.matmul(np.matmul(T[0:n,n:2*n],J),U[0:n,n:2*n])
    ],[np.matmul(np.matmul(U[n:2*n,0:n],K),T[n:2*n,0:n]),U[n:2*n,n:2*n]
    + np.matmul(np.matmul(np.matmul(U[n:2*n,0:n],K),T[n:2*n,n:2*n]),U[0:n,n:2*n])
    ]])
    return S

def c_bas(A,V,h):
    ''' Directly cascading any scattering matrix A (square and with even
    dimensions) with the scattering matrix of a layer of thickness h in which
    the wavevectors are given by V. Since the layer matrix is
    essentially empty, the cascading is much quicker if this is taken
    into account.
    '''
    n=int(A.shape[1]/2)
    D=np.diag(np.exp(1j*V*h))
    S=np.block([[A[0:n,0:n],np.matmul(A[0:n,n:2*n],D)],[np.matmul(D,A[n:2*n,0:n]),np.matmul(np.matmul(D,A[n:2*n,n:2*n]),D)]])
    return S

def marche(a,b,p,n,x):
    '''Computes the Fourier series for a piecewise function having the value
    a over a portion p of the period, starting at position x
    and the value b otherwise. The period is supposed to be equal to 1.
    Division by zero or very small values being not welcome, think about
    not taking round values for the period or for p. Then takes the toeplitz
    matrix generated using the Fourier series.
    '''
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
    '''Attention : a refers to the proportion of e1 in the period, and x0
    to the starting position of this inclusion in a material of
    permittivity e2'''
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

def reseau(k0,a0,pol,e1,e2,n,blocs):
    '''Warning: blocs is a vector with N lines and 2 columns. Each
    line refers to a block of material e2 inside a matrix of material e1,
    giving its size relatively to the period (first column) and its starting
    position.
    Warning : There is nothing checking that the blocks don't overlapp.
    '''
    n_blocs=blocs.shape[0];
    nmod=int(n/2)
    M1=marche(e2,e1,blocs[0,0],n,blocs[0,1])
    M2=marche(1/e2,1/e1,blocs[0,0],n,blocs[0,1])
    if n_blocs>1:
        for k in range(1,n_blocs):
            M1=M1+marche(e2-e1,0,blocs[k,0],n,blocs[k,1])
            M2=M2+marche(1/e2-1/e1,0,blocs[k,0],n,blocs[k,1])
    alpha=np.diag(a0+2*np.pi*np.arange(-nmod,nmod+1))+0j
    if (pol==0):
        M=alpha*alpha-k0*k0*M1
        L,E=np.linalg.eig(M)
        L=np.sqrt(-L+0j)
        L=(1-2*(np.imag(L)<-1e-15))*L
        P=np.block([[E],[np.matmul(E,np.diag(L))]])
    else:
        T=np.linalg.inv(M2)
        M=np.matmul(np.matmul(np.matmul(T,alpha),np.linalg.inv(M1)),alpha)-k0*k0*T
        L,E=np.linalg.eig(M)
        L=np.sqrt(-L+0j)
        L=(1-2*(np.imag(L)<-1e-15))*L
        P=np.block([[E],[np.matmul(np.matmul(M2,E),np.diag(L))]])
    return P,L

def homogene(k0,a0,pol,epsilon,n):
    '''Generates the P matrix and the wavevectors exactly as for a
    periodic layer, just for an homogeneous layer. The results are
    analytic in that case.
    '''
    nmod=int(n/2)
    valp=np.sqrt(epsilon*k0*k0-(a0+2*np.pi*np.arange(-nmod,nmod+1))**2+0j)
    valp=valp*(1-2*(valp<0))
    P=np.block([[np.eye(n)],[np.diag(valp*(pol/epsilon+(1.-pol)))]])
    return P,valp

def interface(P,Q):
    '''Computation of the scattering matrix of an interface, P and Q being the
    matrices given for each layer by homogene, reseau or creneau.
    '''
    n=int(P.shape[1])
    S=np.matmul(np.linalg.inv(np.block([[P[0:n,0:n],-Q[0:n,0:n]],[P[n:2*n,0:n],Q[n:2*n,0:n]]])),np.block([[-P[0:n,0:n],Q[0:n,0:n]],[P[n:2*n,0:n],Q[n:2*n,0:n]]]))
    return S

def psplit(X,lam):
    ''' Envoi de toute lumière incidente dans l'ordre 1.
    '''

    # Structure du vecteur X :
    # 3xN où N est le nombre de blocs, à 1 bloc par couche
    # Hauteur, Largeur, Position :: pour chaque bloc
    # On commence par la période fixe.

    n_layers = len(X)//3
    #print(n_layers)
    X=X.reshape((n_layers,3))
    # On vérifie que c'est constructible...
    for k in range(n_layers-1):
        if min(X[k+1,1]+X[k+1,2],X[k,1]+X[k,2])<max(X[k+1,1],X[k,1]):
            return 100
    d=848.53
    e1=1
    e2=1.5**2

    x=X/d
    l=lam/d
    k0=2*np.pi/l
    nmod=25
    n=2*nmod+1

    pol=0.
    S=np.block([[np.zeros([n,n]),np.eye(n,dtype=np.complex)],[np.eye(n),np.zeros([n,n])]])
    P,V=homogene(k0,0,pol,e1,n)
    V_air = V
    for k in range(0,n_layers):
        bloc = np.array([x[k,1:3]])
        Pc,Vc=reseau(k0,0,pol,e1,e2,n,bloc)
        S=cascade(S,interface(P,Pc))
        S=c_bas(S,Vc,x[k,0])
        P=Pc
        V=Vc
    Pc,Vc=homogene(k0,0,pol,e2,n)
    S=cascade(S,interface(P,Pc))
#    P,V=homogene(k0,0,pol,1,n)
    Te = np.zeros(3)
    Tm = np.zeros(3)
    for k in range(-1,2):
        Te[k+1] = np.abs(S[k+nmod,n+nmod])**2*np.real(V_air[nmod+k]/(k0))
#    print(TE01,Vc[nmod+1])

    pol=1.
    S=np.block([[np.zeros([n,n]),np.eye(n,dtype=np.complex)],[np.eye(n),np.zeros([n,n])]])
    P,V=homogene(k0,0,pol,e1,n)
    for k in range(0,n_layers):
        bloc = np.array([x[k,1:3]])
        Pc,Vc=reseau(k0,0,pol,e1,e2,n,bloc)
        S=cascade(S,interface(P,Pc))
        S=c_bas(S,Vc,x[k,0])
        P=Pc
        V=Vc
    Pc,Vc=homogene(k0,0.,pol,e2,n)
    S=cascade(S,interface(P,Pc))
#    P,V=homogene(k0,0,pol,1,n)
    for k in range(-1,2):
        Tm[k+1] =  np.abs(S[nmod+k,n+nmod])**2*np.real(V_air[nmod+k]/k0)*e2/e1

    cost=1-(Te[2]+Tm[0])/2

    return cost,Te,Tm

def spectre(X):

    N = 100
    lam = np.linspace(400,700,N)
    pol_p = np.zeros(N)
    pol_s = np.zeros(N)
    for k in range(N):

        [loss,Te,Tm] = psplit(X,lam[k])
        pol_s[k] = Te[2]
        pol_p[k] = Tm[0]

    plt.plot(lam,pol_s,lam,pol_p,linewidth = 2)
    plt.show()

def visualization(X,d):
    from matplotlib.patches import Polygon
    import matplotlib.pyplot as plt
    _, ax = plt.subplots(1)

    n_layers = len(X)//3
    X=X.reshape((n_layers,3))
    vert_pos = np.sum(X[:,0]) - np.cumsum(X[:,0])
    for m in range(3):
        for k in range(n_layers):
            A = (X[k,2],vert_pos[k])
            B = (X[k,2],vert_pos[k]+X[k,0])
            C = (X[k,2]+X[k,1],vert_pos[k]+X[k,0])
            D = (X[k,2]+X[k,1],vert_pos[k])
            patch = Polygon([A,B,C,D])
            ax.add_patch(patch)
        X[:,2] = X[:,2]+d
    plt.axis('equal')
    plt.show()


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


n_couches = "4"
algo = "DE"
function = "psplit"
budget = 10000

file_name = f"../ResA/{function}_{algo}_{n_couches}_{budget}.npy"
results = np.load(file_name,allow_pickle = True)


#Graphique pour comparaisons
values = []
bests = []
fig1 = plt.figure(1)

for k in range(len(results)):
    values.append(results[k][1][-1])
    bests.append(results[k][0])
sorted = np.sort(values)
agrum = np.argsort(values)
# Let's sort bests too !
sorted_bests = []
for k in range(len(results)):
    sorted_bests.append(bests[agrum[k]])

plt.plot(sorted)
plt.title(f'Function {function} with {algo}')

# Représenter les courbes de convergence

fig2 = plt.figure(2)
for k in range(len(results)):
    plt.plot(results[k][1])
plt.title("Convergences")


fig3 = plt.figure(3)
X = sorted_bests[0]
#spectre(X)
visualization(X,848.53)
