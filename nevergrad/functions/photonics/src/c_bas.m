% Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
%
% This source code is licensed under the MIT license found in the
% LICENSE file in the root directory of this source tree.
%
% Credit: see CREDIT file in the same folder as this file

function S=c_bas(A,valp,h)
     n=size(A,1)/2;
     D=diag(exp(i*valp*h));
     S=[A(1:n,1:n),A(1:n,n+1:2*n)*D;D*A(n+1:2*n,1:n),D*A(n+1:2*n,n+1:2*n)*D];
end
