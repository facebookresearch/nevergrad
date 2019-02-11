% Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
%
% This source code is licensed under the MIT license found in the
% LICENSE file in the root directory of this source tree.
%
% Credit: see CREDIT file in the same folder as this file

warning("off")
1;

# Computation of the cost function and of the reflectance for a structure beginning with alternating refractive index (beginning with the highest)
# The X vector contains the thicknesses of the layers, beginning with the top layer.
# Providing a wavelength as an extra argument allows to compute the reflectance for this wavelength.

function [cost,R]=chirped(X,varargin)

  bar=length(X);
  if length(varargin)==0
    lambda=linspace(500,800,50);
  else
      lambda=varargin{1};
  end
  theta=0;
  epsilon=[1,3,2];
  m=[1,repmat([2,3],1,bar/2),2];
  hauteur=[0,X,0];
  g=length(m);

  r=zeros(1,length(lambda));
  for j=1:length(lambda)
    k0=2*pi/lambda(j);
    S=[0,1;1,0];
    alpha=sqrt(epsilon(m(1)))*k0*sin(theta);
    gamma(1)=sqrt(epsilon(m(1))*k0^2-alpha^2);
    for k=1:g-1
      gamma(k+1)=sqrt(epsilon(m(k+1))*k0^2-alpha^2);
      u=exp(i*gamma(k)*hauteur(k));
      C=[0,u;u,0];
      S=cascade(S,C);
      C=[gamma(k)/epsilon(m(k))-gamma(k+1)/epsilon(m(k+1)),2*gamma(k+1)/epsilon(m(k+1));2*gamma(k)/epsilon(m(k)),gamma(k+1)/epsilon(m(k+1))-gamma(k)/epsilon(m(k))]/(gamma(k)/epsilon(m(k))+gamma(k+1)/epsilon(m(k+1)));
      S=cascade(S,C);
    endfor
    u=exp(i*gamma(g)*hauteur(g));
    S=cascade(S,[0,u;u,0]);
    r(j)=abs(S(1,1))^2;
  endfor

  R=sum(r)/length(r);
  cost=1-R;

endfunction


X=[];
for i = 1:nargin
  X = [X,str2num(argv(){i})];
endfor
disp(X);
cost = chirped(X);
disp(cost);
