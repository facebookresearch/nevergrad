% Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
%
% This source code is licensed under the MIT license found in the
% LICENSE file in the root directory of this source tree.
%
% Credit: see CREDIT file in the same folder as this file

warning("off")
1;

# Structure of the X vector:
# The first quarter represents the thicknesses of each block (on top first)
# The second represents where each block begins relatively to the beginning of the period
# The third quarter represents the width of each block
# The last quarter represents the thickness of the air layer below each block

2;

function [cost,R]=morpho(X)

# Computation of the efficiencies for the 450 nm wavelength
lambda=449.5897;
theta=0.;
pol=1.;
# Period
d=600.521475;

nmod=25;
n=2*nmod+1;
% Nombre de couches
n_motifs=length(X)/4;
h=X(1:n_motifs)/d;
x0=X(n_motifs+1:2*n_motifs)/d;
a=X(2*n_motifs+1:3*n_motifs)/d;
spacers=X(3*n_motifs+1:4*n_motifs)/d;

# Permittivity of air
e1=1;
# Permittivity of chitin
e2=1.56^2;

l=lambda/d;
k0=2*pi/l;
a0=k0*sin(theta);
[P,V]=homogene(k0,a0,pol,e1,n);
S=[zeros(n),eye(n);eye(n),zeros(n)];
#Computation of the scattering matrix
for j=1:n_motifs
  [Pc,Vc]=creneau(k0,a0,pol,e2,e1,a(j),n,x0(j));
  S=cascade(S,interface(P,Pc));
  S=c_bas(S,Vc,h(j));
  S=cascade(S,interface(Pc,P));
  S=c_bas(S,V,spacers(j));
endfor
[Pc,Vc]=homogene(k0,a0,pol,e2,n);
S=cascade(S,interface(P,Pc));
# Computation of the efficiencies
R=zeros(1,3);
for j=-1:1
  R(j+2)=abs(S(j+nmod+1,nmod+1))^2*V(j+nmod+1)/(k0*sqrt(1-(a0^2)/(k0^2)));
  if (imag(V(j+nmod+1))>0.001)
    R(j+2)=0;
  endif
endfor

R1=R;

# First part of the cost function
cost=1-(R(1)+R(3))/2+R(2)/2;

lambda=[400,500,600,700,800]+0.24587;
l=lambda/d;

bar=0;

for k=1:length(lambda)

k0=2*pi/l(k);
a0=k0*sin(theta);
[P,V]=homogene(k0,a0,pol,e1,n);
# Computation of the scattering matrix
S=[zeros(n),eye(n);eye(n),zeros(n)];
for j=1:n_motifs
  [Pc,Vc]=creneau(k0,a0,pol,e2,e1,a(j),n,x0(j));
  S=cascade(S,interface(P,Pc));
  S=c_bas(S,Vc,h(j));
  S=cascade(S,interface(Pc,P));
  S=c_bas(S,V,spacers(j));
endfor
[Pc,Vc]=homogene(k0,a0,pol,e2,n);
S=cascade(S,interface(P,Pc));
# Computation of the efficiencies
R=zeros(1,3);
for j=-1:1
  R(j+2)=abs(S(j+nmod+1,nmod+1))^2*V(j+nmod+1)/(k0*sqrt(1-(a0^2)/(k0^2)));
  if (imag(V(j+nmod+1))>0.001)
    R(j+2)=0;
  endif
endfor
bar=bar+R(2);
endfor

cost=cost+bar/length(lambda);

# No penalization here
if (cost<0)
  cost=1000;
endif

R=[R1,R(2)];

end



X=[];
for i = 1:nargin
  X = [X,str2num(argv(){i})];
endfor
disp(X);
cost = morpho(X);
disp(cost);
