% Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
%
% This source code is licensed under the MIT license found in the
% LICENSE file in the root directory of this source tree.
%
% Credit: see CREDIT file in the same folder as this file

warning("off")
% Computation of the reflectance of a structure defined by a vector X:
% The first part of the vector contains the permittivities (square of the refractive index) of the different layers
% The second part contains the thicknesses of the layers
% Eventually, a second argument can be provided to compute the reflectance for another wavelength

1;


function [cost,r]=bragg(X,varargin)

  bar=length(X);
  if size(varargin)==0
    lambda=600;
  else
    lambda=varargin{1};
  end
  theta=0;
  epsilon=[1,3,X(1:bar/2)];
  m=[1,[1:bar/2]+2,2];
  hauteur=[0,X(bar/2+1:bar),0];
  k0=2*pi/lambda;
  g=length(m);
  S=[0,1;1,0];
  alpha= sqrt(epsilon(m(1)))*k0*sin(theta);
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
  r=abs(S(1,1))^2;
  cost=1-r;

endfunction

X=[];
for i = 1:nargin
  X = [X,str2num(argv(){i})];
endfor
disp(X);
cost = bragg(X)
disp(cost)
