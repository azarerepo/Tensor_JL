function [varargout]=tensor_randproj_multi_opt(U, option, varargin)

% U: a cell containing mode projection matrices

% option: if set to 'full', random projection is performed on all modes,
% otherwise, if set to 'samples', the last mode is left intact.

% varargin : a series of input tensors

% varargout: the corresponding series of output tensors that have been compressed

% All input tensors MUST have the same sizes

% Ali Zare (zareali@msu.edu). Functions ndim_fold.m and ndim_unfold.m
% from TP Tool (Mathworks) where modified and used.

siz=size(varargin{1});
n_dims=length(siz);
if strcmp(option, 'full')
    N=n_dims;
elseif strcmp(option, 'samples')
    N=n_dims-1; % number of dimensions in each sample
else
    error('Only two options ''full'' and ''samples'' are available.')
end

siz_samp=siz(1:N);
for k=1:N
    
    A=U{k};
    d=size(A,1);
        
    siz_samp(k)=d;
    for i=1:length(varargin)
        
        if k==1
            varargout{i}=varargin{i};
        end
        
        Y_unfold=A*ndim_unfold_me(varargout{i}, k);
                
        if strcmp(option, 'full')
            varargout{i}=ndim_fold_me(Y_unfold, k, siz_samp); % Y, whose mode-n unfolding has been projected
        else
            varargout{i}=ndim_fold_me(Y_unfold, k, [siz_samp, siz(n_dims)]); % Y, whose mode-n unfolding has been projected
        end
        
    end
end


