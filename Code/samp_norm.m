function samp_nrm=samp_norm(T)

% T: input tensor of N+1 dimensions (the last dimension MUST pertain to samples)

% samp_nrm: norm of the samples formed as a row vector

% Ali Zare (zareali@msu.edu)

siz=size(T);
n_dims=length(siz);
n_samp=siz(n_dims); % number of samples (size of the last dimension)

T=reshape(T,[],n_samp);

samp_nrm=zeros(1, n_samp);
for i=1:n_samp
    samp_nrm(i)=norm(T(:,i));
end
