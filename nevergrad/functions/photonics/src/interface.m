% Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
%
% This source code is licensed under the MIT license found in the
% LICENSE file in the root directory of this source tree.
%
% Credit: see CREDIT file in the same folder as this file

function S=interface(P,Q)
n=size(P)(2);
U=[[P(1:n,1:n),-Q(1:n,1:n)];[P(n+1:2*n,1:n),Q(n+1:2*n,1:n)]];
V=[[-P(1:n,1:n),Q(1:n,1:n)];[P(n+1:2*n,1:n),Q(n+1:2*n,1:n)]];
S=inv(U)*V;
end
