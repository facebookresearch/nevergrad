% Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
%
% This source code is licensed under the MIT license found in the
% LICENSE file in the root directory of this source tree.
%
% Credit: see CREDIT file in the same folder as this file

function [P,V]=creneau(k0,a0,pol,e1,e2,a,n,x0)

  nmod=floor(n/2);
  aleph=a0+2*pi*[-nmod:nmod];
  alpha=eye(n,n).*repmat(aleph',1,n);
  if (pol==0)
    M=alpha^2-k0^2*marche(e1,e2,a,n,x0);
  else
    T=inv(marche(1/e1,1/e2,a,n,x0));
    M=T*alpha*inv(marche(e1,e2,a,n,x0))*alpha-k0^2*T;
  end

  [E,L]=eig(M);

  L=sqrt(-L);
  for j=1:n
    if (imag(L(j,j))<0)
      L(j,j)=-L(j,j);
    end
  end

  if (pol==0)
    P=[E;E*L];
  else
    P=[E;marche(1/e1,1/e2,a,n,x0)*E*L];
  end

  V=diag(L);

end
