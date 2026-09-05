function [data, mu, mu_prime] = randomtensor_rankR(I, R, n_samp, option)

% This function generates a rank-R random tensor with a user-defined number of samples.
% The last dimension pertains to samples.

% R: rank of data samples
% n_samp: number of data samples
% I: a vector containing mode sizes for each sample

% mu: maximum modewise coherence
% mu_prime: basis coherence

% Ali Zare (zareali@msu.edu)

n_dims_samp=length(I); % number of dimensions of each data sample
n_dims=n_dims_samp+1; % number of dimensions of the tensor containing data samples

data=[];
mu=zeros(n_samp,n_dims_samp);
mu_prime=zeros(R,R,n_samp);
for i=1:n_samp
    A=cell(1,n_dims_samp);
%     lambda=diag(randn(1,R)); % random weights for rank-1 tensors
    lambda=eye(R); % no weights for rank-1 tensors
    for j=1:n_dims_samp
        if j==1
            if strcmp(option, 'Gaussian')
                
                % low coherence with Gaussians
                A{j}=randn(I(j),R);
                A{j}=A{j}/diag(sqrt(sum((A{j}).^2))); % normalize column norms
                A{j}=A{j}*lambda;
                
            elseif strcmp(option, 'high_coherence')
                
                % high coherence option 1 (low-power gaussian noise added to constant, with random column signs)
                A{j}=(ones(I(j),R)+sqrt(0.1)*randn(I(j),R))*diag(2*(rand(1,R)<0.5)-1);
                A{j}=A{j}/diag(sqrt(sum((A{j}).^2))); % normalize column norms
                A{j}=A{j}*lambda;
                
%                 % high coherence option 2 (uniform between a and b, with random column signs)
%                 a=0.8; b=1; 
%                 A{j}=((b-a)*(rand(I(j),R)+a))*diag(2*(rand(1,R)<0.5)-1);
%                 A{j}=A{j}/diag(sqrt(sum((A{j}).^2))); % normalize column norms
%                 A{j}=A{j}*lambda;
            end
        else
            if strcmp(option, 'Gaussian')
                
                % low coherence with Gaussians
                A{j}=randn(I(j),R);
                A{j}=A{j}/diag(sqrt(sum((A{j}).^2))); % normalize column norms
                
            elseif strcmp(option, 'high_coherence')
                
                % high coherence option 1 (low-power gaussian noise added to constant, with random column signs)
                A{j}=(ones(I(j),R)+sqrt(0.1)*randn(I(j),R))*diag(2*(rand(1,R)<0.5)-1);
                A{j}=A{j}/diag(sqrt(sum((A{j}).^2))); % normalize column norms
                
%                 % high coherence option 2 (uniform between a and b, with random column signs)
%                 b=1; a=0.8;
%                 A{j}=((b-a)*(rand(I(j),R)+a))*diag(2*(rand(1,R)<0.5)-1);
%                 A{j}=A{j}/diag(sqrt(sum((A{j}).^2))); % normalize column norms
            end
        end
        G=A{j}'*A{j}; % Gram matrix
        for k=1:R
            G(k,k)=0;
        end
        mu(i,j)=max(max(abs(G))); % modewise coherence for mode j of sample i
    end
    
    if R>1
        for k=1:R-1
            for h=k+1:R
                mu_prime(k,h,i)=1;
                for j=1:n_dims_samp
                    mu_prime(k,h,i)=mu_prime(k,h,i)*abs(sum(A{j}(:,k).*A{j}(:,h))); % Basis coherence for sample i
                end
            end
        end
    else
        mu_prime=1; % this is just a number. Coherence does not make sense if R=1.
    end
    
    data = cat(n_dims, data, cpd_rankone_reconst(A));
end