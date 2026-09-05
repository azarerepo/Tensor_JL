% Implementation of the JL experiments for norm in:
% M.A. Iwen, Deanna Needell, Elizaveta Rebrova, Ali Zare, "Lower Memory
% Oblivious (Tensor) Subspace Embeddings with Fewer Random Bits: Modewise
% Methods for Least Squares", SIAM Journal on Matrix Analysis  and
% Applications, Vol. 42-1, pages 376 -- 416, 2021.

% Implemented by Ali Zare (zareali@msu.edu)

% JL embedding on MRI data (3 samples, ADNI data)
% Comparison between 1-stage modewise JL (Gaussian and RFD) and 2-stage JL
% (Gaussian+Gaussian), (Gaussian+RFD), (RFD + RFD) and (vectorized+RFD)

% In RFD, fft() is used instead of the DFT matrix in the vectorized cases,
% i.e., in
% the 2nd stage JL and the vectorized JL.
% In modewise RFD, random rows of the DFT matrix are used instead of the
% RF part of RFD.


clear

% =============================================================
%%% inputs
% =============================================================
in_data_type='real'; % 'real' or 'synth'

%%% Synthetic data
% Random data specifications
n_samp=10; % number of data samples
I=100*ones(1,4); % dimensions of each sample
R=10; % rank of data samples
option='high_coherence'; % 'high_coherence' or 'Gaussian'

%%% Other specifiations
% dim_frac=[0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.15]; % for MRI data
dim_frac=[0.05 0.1 0.15]; % for quick testing
dim_frac_sec=0.05; % fraction used in the 2nd JL

N_trials=20; % number of trials (1000 in the paper)

% =============================================================
% =============================================================
t_start=tic;

if strcmp(in_data_type,'synth')
    fprintf('\nGenerating random input data ... \n')
    n_dims_samp=length(I); % number of dimensions of each data sample
    n_dims=n_dims_samp+1; % number of dimensions of the tensor containing data samples
    
    %%% data generation (rank R tensor)
    [data, mu, mu_prime]=randomtensor_rankR(I, R, n_samp, option);
    
    %%%% Save it so it doesn't have to be loaded each time
    % save(sprintf('Tensor_Rank%d_opt-%s.mat', R, option), 'data') % save data
    
elseif strcmp(in_data_type,'real')
    fprintf('\nLoading input data ... \n')
        
    load('SIMAX_3sample_MRI.mat')
    fprintf('\nInput data loaded.\n')
    n_dims=length(size(data));
    n_samp=size(data,n_dims);
end
fprintf('\nInput data loaded/generated.\n')
siz=size(data);
siz_samp=prod(siz(1:n_dims-1)); % sample size

%%% norm of the samples
norm_orig=samp_norm(data);

norm_proj_1st=zeros(N_trials, n_samp, length(dim_frac));
norm_proj_F_1st=zeros(N_trials, n_samp, length(dim_frac));
norm_proj=zeros(N_trials, n_samp, length(dim_frac));
norm_proj_F=zeros(N_trials, n_samp, length(dim_frac));
norm_proj_FF=zeros(N_trials, n_samp, length(dim_frac));
norm_proj_vec=zeros(N_trials, n_samp, length(dim_frac));

matgen_modewise_time=zeros(N_trials,length(dim_frac),n_dims-1); % for modewise Gaussian matrices
matgen_2nd_time=zeros(N_trials,length(dim_frac));

rand_proj_1st_time=zeros(N_trials, length(dim_frac));
rand_proj_F_1st_time=zeros(N_trials, length(dim_frac));
rand_proj_time=zeros(N_trials, 4, length(dim_frac));

randbit_modewise_time=zeros(N_trials,length(dim_frac),N_trials);
randrest_modewise_time=zeros(N_trials,length(dim_frac),N_trials);
matgen_F_modewise_time=zeros(N_trials,length(dim_frac),N_trials);
randbit_2nd_time=zeros(N_trials,length(dim_frac));
randbit_vec_time=zeros(N_trials,length(dim_frac));
randrest_2nd_time=zeros(N_trials,length(dim_frac));
randrest_vec_time=zeros(N_trials,length(dim_frac));

d_sec=zeros(1,length(dim_frac));
vec_target_dim=zeros(1,length(dim_frac));
for p=1:length(dim_frac)
    
    d=zeros(1,n_dims-1);
    vec_target_dim(p)=1; % 1st JL target dim after vectorization
    for kk=1:n_dims-1
        d(kk)=ceil(dim_frac(p)*siz(kk));
        vec_target_dim(p)=vec_target_dim(p)*d(kk);
    end
    d_sec(p)=ceil(dim_frac_sec*vec_target_dim(p)); % 2nd JL target dimension
    
    for k=1:N_trials
        fprintf('\ndim_frac(%d) \t Trial %d \n', p, k)
        t1=tic;
        
        % 1st JL parameters
        A=cell(1,n_dims);
        A_F=cell(1,n_dims);
        D=cell(1,n_dims);
        R_rand_idx=cell(1,n_dims);
        for kk=1:n_dims-1
            tic
            A{kk}=randn(d(kk), siz(kk))/sqrt(d(kk));
            matgen_modewise_time(k,p,kk)=toc;   
            
            tic
            D{kk}=2*(rand(1, siz(kk))<0.5)-1;
            randbit_modewise_time(k,p,kk)=toc;
            tic
            R_rand_idx{kk}=floor(siz(kk)*rand(1,d(kk))); % random restriction with replacement
            randrest_modewise_time(k,p,kk)=toc;
            tic
            A_F{kk}=exp(-1i*2*pi/siz(kk)*(R_rand_idx{kk}).'*(0:siz(kk)-1)).*D{kk}/sqrt(d(kk));
            matgen_F_modewise_time(k,p,kk)=toc;
        end
                
        % 2nd JL parameters
        tic
        A_2nd=randn(d_sec(p), vec_target_dim(p))/sqrt(d_sec(p));
        matgen_2nd_time(k,p)=toc;
        
        tic
        R_rand_idx_sec=floor(vec_target_dim(p)*rand(1,d_sec(p))); % random restriction with replacement
        randrest_2nd_time(k,p)=toc;
        tic
        D_sec=2*(rand(1, vec_target_dim(p))<0.5)-1;
        randbit_2nd_time(k,p)=toc;
        
        %===============================================================================
        % Gaussian + RFD
        tic
        samp1_proj=tensor_randproj_multi_opt(A, 'samples', data); % Gaussian JL
        rand_proj_1st_time(k,p)=toc;
        norm_proj_1st(k,:,p)=samp_norm(samp1_proj);
        
        % 2nd JL: RFD on the vectorized data
        tic
        samp1_proj_F=fft(sparse(1:vec_target_dim(p), 1:vec_target_dim(p), D_sec)*reshape(samp1_proj, [], n_samp)); % 2nd JL: RFD
        samp1_proj_F=samp1_proj_F(R_rand_idx_sec+1,:)/sqrt(d_sec(p));
        rand_proj_time(k,2,p)=toc;
        norm_proj_F(k,:,p)=samp_norm(samp1_proj_F);
        
        %===============================================================================
        % Gaussian + Gaussian
        
        % 1st JL: already done above
        
        % 2nd JL: Gaussian JL on the vectorized data after the 1st JL
        tic
        samp1_proj=A_2nd*reshape(samp1_proj, [], n_samp);
        rand_proj_time(k,1,p)=toc;
        norm_proj(k,:,p)=samp_norm(samp1_proj);
        
        %===============================================================================
        % RFD + RFD
        tic
        samp1_proj_F1=tensor_randproj_multi_opt(A_F, 'samples', data); % RFD
        rand_proj_F_1st_time(k,p)=toc;
        norm_proj_F_1st(k,:,p)=samp_norm(samp1_proj_F1);
        
        % 2nd JL: RFD on the vectorized data
        tic
        samp1_proj_FF=fft(sparse(1:vec_target_dim(p), 1:vec_target_dim(p), D_sec)*reshape(samp1_proj_F1, [], n_samp)); % 2nd JL: RFD
        samp1_proj_FF=samp1_proj_FF(R_rand_idx_sec+1,:)/sqrt(d_sec(p));
        rand_proj_time(k,3,p)=toc;
        norm_proj_FF(k,:,p)=samp_norm(samp1_proj_FF);
        
        %===============================================================================
        % RFD after vectorization
        tic
        R_rand_idx_vec=floor(siz_samp*rand(1,d_sec(p))); % random restriction with replacement
        randrest_vec_time(k,p)=toc;
        tic
        D_vec=2*(rand(1, siz_samp)<0.5)-1;
        randbit_vec_time(k,p)=toc;
        
        tic
        samp1_proj_vec=fft(sparse(1:siz_samp, 1:siz_samp, D_vec)*reshape(data, [], n_samp));
        samp1_proj_vec=samp1_proj_vec(R_rand_idx_vec+1,:)/sqrt(d_sec(p));
        rand_proj_time(k,4,p)=toc;
        norm_proj_vec(k,:,p)=samp_norm(samp1_proj_vec);
        
        
        t2=toc(t1);
        fprintf('Elapsed time: %f \n', t2)
        
        
    end
    
    
end
end_time=toc(t_start);

%%% save results
% if strcmp(in_data_type,'synth')
%     save(sprintf('results_JL_new4_%s_R%d_opt-%s_Ntrials%d_dimfrac%.2f_dimfracsec%.2f.mat', in_data_type, R, option, N_trials, dim_frac(1), dim_frac_sec),...
%         'norm_orig', 'norm_proj_1st', 'norm_proj_F_1st', 'norm_proj', 'norm_proj_F','norm_proj_FF', 'norm_proj_vec',...
%         'I', 'R', 'mu', 'mu_prime', 'dim_frac', 'dim_frac_sec', 'N_trials', 'n_samp', 'end_time', 'in_data_type', 'option', 'n_dims',...
%         'matgen_modewise_time','rand_proj_1st_time',...
%         'd_sec','matgen_2nd_time','randbit_2nd_time','randrest_2nd_time','rand_proj_time',...
%         'randbit_vec_time','randrest_vec_time',...
%         'rand_proj_1st_time','rand_proj_F_1st_time',...
%         'randbit_modewise_time','randrest_modewise_time','matgen_F_modewise_time',...
%         'siz_samp','d_sec','vec_target_dim')
% else
%     save(sprintf('results_JL_new4_%s_Ntrials%d_dimfrac%.2f_dimfracsec%.2f.mat', in_data_type, N_trials, dim_frac(1), dim_frac_sec),...
%         'norm_orig', 'norm_proj_1st', 'norm_proj_F_1st', 'norm_proj', 'norm_proj_F','norm_proj_FF', 'norm_proj_vec',...
%         'dim_frac', 'dim_frac_sec', 'N_trials', 'n_samp', 'end_time', 'in_data_type', 'option', 'n_dims', ...
%         'matgen_modewise_time','rand_proj_1st_time',...
%         'd_sec','matgen_2nd_time','randbit_2nd_time','randrest_2nd_time','rand_proj_time',...
%         'randbit_vec_time','randrest_vec_time',...
%         'rand_proj_1st_time','rand_proj_F_1st_time',...
%         'randbit_modewise_time','randrest_modewise_time','matgen_F_modewise_time',...
%         'siz_samp','d_sec','vec_target_dim')
% end



%%%===========================================================================
%%%===========================================================================
% %%%% plot results
rand_proj_1st_time_mean=mean(rand_proj_1st_time);
time_1st_mean=mean(sum(matgen_modewise_time,n_dims-1))+rand_proj_1st_time_mean;
rand_proj_F_1st_time_mean=mean(rand_proj_F_1st_time);
time_F_1st_mean=mean(sum(matgen_F_modewise_time,n_dims-1))+...
    mean(sum(randbit_modewise_time,n_dims-1))+mean(sum(randrest_modewise_time,n_dims-1))+...
    rand_proj_F_1st_time_mean;

rand_proj_time_mean=mean(rand_proj_time);
rand_proj_time_mean=reshape(rand_proj_time_mean, 4, length(dim_frac));
time_mean(1,:)=time_1st_mean+rand_proj_time_mean(1,:)+mean(matgen_2nd_time); % Gaussian + Gaussian
time_mean(2,:)=time_1st_mean+...
    mean(randbit_2nd_time)+rand_proj_time_mean(2,:)+mean(randrest_2nd_time); % Gaussian + RFD
time_mean(3,:)=time_F_1st_mean+...
    mean(randbit_2nd_time)+rand_proj_time_mean(3,:)+mean(randrest_2nd_time); % RFD + RFD
time_mean(4,:)=mean(mean(randbit_vec_time))+rand_proj_time_mean(4,:)+mean(randrest_vec_time); % vectorize + RFD



% plot results for time
figure,
loglog(d_sec/siz_samp,time_mean(1,:),'-*',d_sec/siz_samp,time_mean(2,:),'-s',...
    d_sec/siz_samp,time_mean(3,:),'-d',d_sec/siz_samp,time_mean(4,:),'-x','LineWidth',1.4,'MarkerSize',10)
legend('Gaussian+Gaussian','Gaussian+RFD','RFD+RFD','vec+RFD','FontSize',14,'Location','Best')
xlabel('$c_{tot}$', 'Interpreter','latex','FontSize',18)
ylabel('$t$ (s)','Interpreter','latex', 'FontSize', 14)
ylh = get(gca,'ylabel');
set(ylh, 'Rotation', 0, 'VerticalAlignment', 'middle', 'HorizontalAlignment', 'right')
title('Runtime','Interpreter','latex', 'FontSize', 16)
grid on

figure,
loglog(vec_target_dim/siz_samp,time_1st_mean,'-o',vec_target_dim/siz_samp,time_F_1st_mean,'-p',...
    d_sec/siz_samp,time_mean(1,:),'-*',d_sec/siz_samp,time_mean(2,:),'-s',...
    d_sec/siz_samp,time_mean(3,:),'-d',d_sec/siz_samp,time_mean(4,:),'-x','LineWidth',1.4,'MarkerSize',10)
legend('Gaussian','RFD','Gaussian+Gaussian','Gaussian+RFD','RFD+RFD','vec+RFD','FontSize',14,'Location','Best')
xlabel('$c_{tot}$', 'Interpreter','latex','FontSize',18)
ylabel('$t$ (s)','Interpreter','latex', 'FontSize', 14)
ylh = get(gca,'ylabel');
set(ylh, 'Rotation', 0, 'VerticalAlignment', 'middle', 'HorizontalAlignment', 'right')
title('Runtime','Interpreter','latex', 'FontSize', 16)
grid on


%%% plot results for norm
result_norm_ndim=length(size(norm_proj));

norm_proj_1st_mean=mean(norm_proj_1st);
norm_proj_1st_mean=reshape(norm_proj_1st_mean, n_samp, length(dim_frac));
norm_proj_1st_std=std(norm_proj_1st, 0, 1);

norm_proj_F_1st_mean=mean(norm_proj_F_1st);
norm_proj_F_1st_mean=reshape(norm_proj_F_1st_mean, n_samp, length(dim_frac));
norm_proj_F_1st_std=std(norm_proj_F_1st, 0, 1);

norm_proj_mean=mean(norm_proj);
norm_proj_mean=reshape(norm_proj_mean, n_samp, length(dim_frac));
norm_proj_std=std(norm_proj, 0, 1);

norm_proj_F_mean=mean(norm_proj_F);
norm_proj_F_mean=reshape(norm_proj_F_mean, n_samp, length(dim_frac));
norm_proj_F_std=std(norm_proj_F, 0, 1);

norm_proj_FF_mean=mean(norm_proj_FF);
norm_proj_FF_mean=reshape(norm_proj_FF_mean, n_samp, length(dim_frac));
norm_proj_FF_std=std(norm_proj_FF, 0, 1);

norm_proj_vec_mean=mean(norm_proj_vec);
norm_proj_vec_mean=reshape(norm_proj_vec_mean, n_samp, length(dim_frac));
norm_proj_vec_std=std(norm_proj_F, 0, 1);

norm_proj_1st_mn=zeros(1, length(dim_frac));
norm_proj_F_1st_mn=zeros(1, length(dim_frac));
norm_proj_mn=zeros(1, length(dim_frac));
norm_proj_F_mn=zeros(1, length(dim_frac));
norm_proj_FF_mn=zeros(1, length(dim_frac));
norm_proj_vec_mn=zeros(1, length(dim_frac));
lgnd=cell(1,length(dim_frac));
for k=1:length(dim_frac)
    norm_proj_1st_mn_r=norm_proj_1st_mean(:,k);
    norm_proj_1st_mn_r=norm_proj_1st_mn_r.';
    norm_proj_1st_mn(k)=mean(norm_proj_1st_mn_r./norm_orig);
    
    norm_proj_F_1st_mn_r=norm_proj_F_1st_mean(:,k);
    norm_proj_F_1st_mn_r=norm_proj_F_1st_mn_r.';
    norm_proj_F_1st_mn(k)=mean(norm_proj_F_1st_mn_r./norm_orig);
    
    norm_proj_mn_r=norm_proj_mean(:,k);
    norm_proj_mn_r=norm_proj_mn_r.';
    norm_proj_mn(k)=mean(norm_proj_mn_r./norm_orig);
    
    norm_proj_F_mn_r=norm_proj_F_mean(:,k);
    norm_proj_F_mn_r=norm_proj_F_mn_r.';
    norm_proj_F_mn(k)=mean(norm_proj_F_mn_r./norm_orig);
    
    norm_proj_FF_mn_r=norm_proj_FF_mean(:,k);
    norm_proj_FF_mn_r=norm_proj_FF_mn_r.';
    norm_proj_FF_mn(k)=mean(norm_proj_FF_mn_r./norm_orig);
    
    norm_proj_vec_mn_r=norm_proj_vec_mean(:,k);
    norm_proj_vec_mn_r=norm_proj_vec_mn_r.';
    norm_proj_vec_mn(k)=mean(norm_proj_vec_mn_r./norm_orig);
    
end
figure,
semilogx(d_sec/siz_samp, norm_proj_mn, '*', 'MarkerSize', 12, 'LineWidth', 1.4)
hold on
semilogx(d_sec/siz_samp, norm_proj_F_mn, '*', 'MarkerSize', 12, 'LineWidth', 1.4)
hold on
semilogx(d_sec/siz_samp, norm_proj_FF_mn, '*', 'MarkerSize', 12, 'LineWidth', 1.4)
hold on
semilogx(d_sec/siz_samp, norm_proj_vec_mn, '*', 'MarkerSize', 12, 'LineWidth', 1.4)
xlabel('$c_{tot}$', 'Interpreter', 'latex', 'FontSize', 18)
ylabel('$c_{n,\mathcal{X}}$','Interpreter','latex', 'FontSize', 18)
ylh = get(gca,'ylabel');
set(ylh, 'Rotation', 0, 'VerticalAlignment', 'middle', 'HorizontalAlignment', 'right')
title(sprintf('Relative norm averaged over %d samples in %d trials.', n_samp, N_trials),'Interpreter','latex', 'FontSize', 16)
legend('Gaussian+Gaussian','Gaussian+RFD','RFD+RFD','vec+RFD','FontSize',14,'Location','Best')
grid on

figure,
semilogx(vec_target_dim/siz_samp, norm_proj_1st_mn, '-o', 'MarkerSize', 12, 'LineWidth', 1.4)
hold on
semilogx(vec_target_dim/siz_samp, norm_proj_F_1st_mn, '-p', 'MarkerSize', 12, 'LineWidth', 1.4)
hold on
semilogx(d_sec/siz_samp, norm_proj_mn, '-*', 'MarkerSize', 12, 'LineWidth', 1.4)
hold on
semilogx(d_sec/siz_samp, norm_proj_F_mn, '-s', 'MarkerSize', 12, 'LineWidth', 1.4)
hold on
semilogx(d_sec/siz_samp, norm_proj_FF_mn, '-d', 'MarkerSize', 12, 'LineWidth', 1.4)
hold on
semilogx(d_sec/siz_samp, norm_proj_vec_mn, '-x', 'MarkerSize', 12, 'LineWidth', 1.4)
xlabel('$c_{tot}$', 'Interpreter', 'latex', 'FontSize', 18)
ylabel('$c_{n,\mathcal{X}}$','Interpreter','latex', 'FontSize', 18)
ylh = get(gca,'ylabel');
set(ylh, 'Rotation', 0, 'VerticalAlignment', 'middle', 'HorizontalAlignment', 'right')
title(sprintf('Relative norm averaged over %d samples in %d trials.', n_samp, N_trials),'Interpreter','latex', 'FontSize', 16)
legend('Gaussian','RFD','Gaussian+Gaussian','Gaussian+RFD','RFD+RFD','vec+RFD','FontSize',14,'Location','Best')
grid on

