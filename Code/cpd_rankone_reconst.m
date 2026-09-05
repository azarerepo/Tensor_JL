function T = cpd_rankone_reconst(U)
 
% Reconstruction of a tensor as the sum of R rank-1 tensors
% T: reconstructed rank-1 tensor
% U: a cell containing factor matrices along each mode forming rank-1 tensors
% All elements of U must have the same number of columns

% Ali Zare (zareali@msu.edu). The function ndim_expand.m from TP Tool
% (Mathworks) was used.

n_dims=length(U); % number of modes of the reconstructed rank-1 tensor

R=size(U{1},2); % number of rank-1 tensors

for k=2:n_dims
    if size(U{k},2)~=R
        error('All elements of U must have the same number of columns.')
    end
end

T=0;
for r=1:R
    T1=U{1}(:,r);
    for d=2:n_dims
        T1=ndim_expand(T1,U{d}(:,r));
    end
    T=T+T1;
end
