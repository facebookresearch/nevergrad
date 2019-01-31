% Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
%
% This source code is licensed under the MIT license found in the
% LICENSE file in the root directory of this source tree.
%
% Credit: see CREDIT file in the same folder as this file

function T=marche(a,b,p,n,x0)
l=i*(a-b)./(2*pi*[0:n-1]).*(exp(-2*i*pi*p*[0:n-1])-1).*exp(-2*i*pi*[0:n-1]*x0);
m=i*(b-a)./(2*pi*[0:n-1]).*(exp(2*i*pi*p*[0:n-1])-1).*exp(2*i*pi*[0:n-1]*x0);
l(1)=p*a+(1-p)*b;
m(1)=p*a+(1-p)*b;
T=toeplitz(l,m);
end
