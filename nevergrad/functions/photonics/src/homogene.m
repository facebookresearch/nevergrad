% Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
%
% This source code is licensed under the MIT license found in the
% LICENSE file in the root directory of this source tree.
%
% Credit: see CREDIT file in the same folder as this file

function [P,valp]=homogene(k0,a0,pol,epsilon,n)

nmod=floor(n/2);
valp=sqrt(epsilon*k0^2-(a0+2*pi*[-nmod:nmod]).^2);

for j=1:n
  if (imag(valp(j))<0)
    valp(j)=-valp(j);
  end
end

if (pol==0)
  P=[eye(n,n);eye(n,n).*repmat(valp.',1,n)];
else
  P=[eye(n,n);eye(n,n).*repmat(valp.',1,n)/epsilon];
end
end
