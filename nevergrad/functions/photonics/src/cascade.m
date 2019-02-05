% Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
%
% This source code is licensed under the MIT license found in the
% LICENSE file in the root directory of this source tree.
%
% Credit: see CREDIT file in the same folder as this file

function S=cascade(T,U)
n=size(T)(1)/2;
A=T(1:n,1:n);
B=T(1:n,n+1:2*n);
C=T(n+1:2*n,1:n);
D=T(n+1:2*n,n+1:2*n);
E=U(1:n,1:n);
F=U(1:n,n+1:2*n);
G=U(n+1:2*n,1:n);
H=U(n+1:2*n,n+1:2*n);
J=inv(eye(n,n)-E*D);
K=inv(eye(n,n)-D*E);
S=[[A+B*J*E*C,B*J*F];[G*K*C,H+G*K*D*F]];
end
