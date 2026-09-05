function T = ndim_fold_me(M, dim, siz)
%NDIM_FOLD Restore a matrix into a multidimensional array with the given size
%	T = NDIM_FOLD(M, dim, siz)
%	
%	M   - matrix
%	dim - the columns of M will turn into this dimension of T
%	siz - size(T) (prod(siz)==numel(M), siz(dim)==size(M,1) should hold)
%
%	T   - multidimensional array containing the column vectors of M along dim dimension
%
%	eg. ndim_fold([2*ones(3) 3*ones(3)], 1, [3 3 2])
%
%	See also NDIM_UNFOLD_ME.

%   Modified version of ndim_fold.m from TP Tool (Mathworks)
%   Modified to adjust to moving through modes form lower to higher in ndim_unfold_me.m, and also prevent wrong inputs.
%   Modified by Ali Zare (zareali@msu.edu)

if prod(siz)~=numel(M) || siz(dim)~=size(M,1)
    error('Conditions prod(siz)~=numel(M) and siz(dim)~=size(M,1) should hold.')
end

ndim = length(siz);

if ndim == 2
	% T is 1D or 2D
	if dim == 2
		T = M';
	else
		T = M;
	end
else
	
	% restore M into T
    new_size=siz;
    new_size(dim)=[];
    new_size=[size(M,1), new_size]; 
    T = reshape(M, new_size);
    	
	% rotate T into the right position
	dim_order=1:ndim;
    dim_order(dim)=[];
    dim_order=[dim, dim_order];
    [~, dim_order]=sort(dim_order, 'ascend');
	T = permute(T, dim_order);
end
