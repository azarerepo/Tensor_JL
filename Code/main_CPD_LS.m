% Implementation of the JL experiments for Compressed Least Squares in:
% M.A. Iwen, Deanna Needell, Elizaveta Rebrova, Ali Zare, "Lower Memory
% Oblivious (Tensor) Subspace Embeddings with Fewer Random Bits: Modewise
% Methods for Least Squares", SIAM Journal on Matrix Analysis  and
% Applications, Vol. 42-1, pages 376 -- 416, 2021.

% Implemented by Ali Zare (zareali@msu.edu)

% CPD error vs rank
% Comparison of compressed Least Squares using 2-stage JL with compressed LS using vectorized data.
% The 2-stage JL is done in two ways: (Gaussian + RFD) and (RFD + RFD).
% The 1-stage JL on vectorized data is done using RFD.
% In the 2nd stage and the vectorized case fft() is used in RFD.
% In the 1ns stage, the DFT matrix is used in RFD.

% Error measure: as defined in the paper.

clear

% =============================================================
%%% inputs
% =============================================================
% modewise compression
% dim_frac=[0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.12 0.15 0.18]; % in the paper
dim_frac=[0.03 0.1]; % for quick testing
dim_frac_sec=0.05;

N_trials=10; % number of trials (100 in the paper)

r=5:5:205; % CPD-rank (in this code, it is fixed for the CPD factor matrices loaded below)

r_sel=[10 40]; % selected ranks from r above to use for experiments, [40 75 110] in the paper

% =============================================================
% =============================================================

fprintf('\nLoading input data ... \n')
load('threesample_MRI.mat')
fprintf('\nInput data loaded.\n')
n_dims=length(size(data));
n_dims_samp=n_dims-1;
n_samp=size(data,n_dims);
data=data(:,:,:,1); % 1st sample of the 3 MRI samples was used in the paper experiments

fprintf('\nInput data loaded.\n')

%%% Loading CPD factor matrices
fprintf('\nLoading CPD factor matrices...\n')
load('MRI_samp10_CPD_r5to205.mat')
fprintf('\nCPD factor matrices loaded.\n')

%%% ==================================================================================
%%% Compressed Least Squares for CP
%%% ==================================================================================
idx=zeros(1,length(r_sel));
for k=1:length(r_sel)
    idx(k)=find(r==r_sel(k));
end

t_start=tic;

% To reconstruct rank-1 tensors from CPD for chosen ranks:
U1=U(idx);

alpha_LS=cell(1,length(U1));
alpha_proj_LS_norm=zeros(N_trials, length(dim_frac), length(U1));
alpha_proj_F_LS_norm=zeros(N_trials, length(dim_frac), length(U1));
alpha_proj_vec_LS_norm=zeros(N_trials, length(dim_frac), length(U1));

matgen_modewise_time=zeros(N_trials,length(dim_frac),n_dims-1);
matgen_F_modewise_time=zeros(N_trials,length(dim_frac),n_dims-1);
rand_proj_1st_time=zeros(N_trials, 2, length(dim_frac));
rand_proj_time=zeros(N_trials, 3, length(dim_frac));
randbit_2nd_time=zeros(N_trials,length(dim_frac));
randbit_vec_time=zeros(N_trials,length(dim_frac));
randrest_2nd_time=zeros(N_trials,length(dim_frac));
randrest_vec_time=zeros(N_trials,length(dim_frac));
randbit_modewise_time=zeros(N_trials,length(dim_frac),n_dims-1);
randrest_modewise_time=zeros(N_trials,length(dim_frac),n_dims-1);

LS_sol_1st_time=zeros(N_trials, 2, length(dim_frac));
LS_sol_time=zeros(N_trials, 3, length(dim_frac));
d_sec=zeros(1,length(dim_frac));
vec_target_dim=zeros(1,length(dim_frac));

err_true=zeros(1, length(U1));
err_proj_1st=zeros(N_trials, length(dim_frac), 2, length(U1));
err_proj=zeros(N_trials, length(dim_frac), 3, length(U1));
for m=1:length(U1)
    
    r1_data=[];
    for p=1:r(idx(m))
        B=cell(1,n_dims_samp);
        for q=1:n_dims_samp
            B{q}=U1{m}{q}(:,p);
        end
        r1_data=cat(n_dims_samp+1, r1_data, cpd_rankone_reconst(B));
    end
    
    r1_data_vec=reshape(r1_data,[],size(r1_data, length(size(r1_data))));
    
    data_vec=data(:);
    
    alpha_LS{m}=(r1_data_vec'*r1_data_vec)\(r1_data_vec'*data_vec); % true solution
    
    err_true(m)=norm(data_vec-r1_data_vec*alpha_LS{m});
    
    
    r1_data=cat(n_dims_samp+1, r1_data, data);
    s.subs{n_dims}=size(r1_data, n_dims);
    siz=size(r1_data);
    for p=1:length(dim_frac)
        
        d=zeros(1,n_dims-1);
        vec_target_dim(p)=1;
        for kk=1:n_dims-1
            d(kk)=ceil(dim_frac(p)*siz(kk));
            vec_target_dim(p)=vec_target_dim(p)*d(kk);
        end
        d_sec(p)=ceil(dim_frac_sec*vec_target_dim(p));
        
        for k=1:N_trials
            
            fprintf('\nrank %d \t dim_frac(%d) \t Trial %d \n',r(idx(m)), p, k)
            t1=tic;
            
            % 1st JL parameters
            A=cell(1,n_dims-1);
            A_F=cell(1,n_dims-1);
            D=cell(1,n_dims-1);
            R_rand_idx=cell(1,n_dims-1);
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
            R_rand_idx_sec=floor(vec_target_dim(p)*rand(1,d_sec(p)));
            randrest_2nd_time(k,p)=toc;
            tic
            D_sec=2*(rand(1, vec_target_dim(p))<0.5)-1;
            randbit_2nd_time(k,p)=toc;
            
            %===============================================================================
            %%% Gaussian + RFD
            tic
            r1_data_proj=tensor_randproj_multi_opt(A, 'samples', r1_data); % Gaussian JL
            rand_proj_1st_time(k,1,p)=toc;
            x=reshape(r1_data_proj,[],r(idx(m))+1);
            tic
            alpha_proj_1st_LS=x(:,1:end-1)\x(:,end);
            LS_sol_1st_time(k,1,p)=toc;
            err_proj_1st(k,p,1,m)=norm(data_vec-r1_data_vec*alpha_proj_1st_LS);
            
            tic
            r1_data_proj_vec=fft(sparse(1:vec_target_dim(p), 1:vec_target_dim(p), D_sec)*reshape(r1_data_proj,[],r(idx(m))+1)); % 2nd JL: RFD
            r1_data_proj_vec=r1_data_proj_vec(R_rand_idx_sec+1,:)/sqrt(d_sec(p));
            rand_proj_time(k,1,p)=rand_proj_1st_time(k,1,p)+toc;
            
            sample_data_proj_vec=r1_data_proj_vec(:,end);
            r1_data_proj_vec(:,end)=[];
            
            tic
            alpha_proj_LS=r1_data_proj_vec\sample_data_proj_vec;
            LS_sol_time(k,1,p)=toc;
            
            err_proj(k,p,1,m)=norm(data_vec-r1_data_vec*alpha_proj_LS);
            
            alpha_proj_LS_norm(k,p,m)=norm(alpha_proj_LS);
            
            %===============================================================================
            %%% RFD + RFD
            tic
            r1_data_proj_F=tensor_randproj_multi_opt(A_F, 'samples', r1_data); % fast JL: RFD
            rand_proj_1st_time(k,2,p)=toc;
            x=reshape(r1_data_proj_F,[],r(idx(m))+1);
            tic
            alpha_proj_F_1st_LS=x(:,1:end-1)\x(:,end);
            LS_sol_1st_time(k,2,p)=toc;
            err_proj_1st(k,p,2,m)=norm(data_vec-r1_data_vec*alpha_proj_F_1st_LS);
            
            tic
            r1_data_proj_F_vec=fft(sparse(1:vec_target_dim(p), 1:vec_target_dim(p), D_sec)*reshape(r1_data_proj_F, [], r(idx(m))+1)); % 2nd JL: RFD
            r1_data_proj_F_vec=r1_data_proj_F_vec(R_rand_idx_sec+1,:)/sqrt(d_sec(p));
            rand_proj_time(k,2,p)=rand_proj_1st_time(k,2,p)+toc;
            
            sample_data_proj_F_vec=r1_data_proj_F_vec(:,end);
            r1_data_proj_F_vec(:,end)=[];
            
            tic
            alpha_proj_F_LS=r1_data_proj_F_vec\sample_data_proj_F_vec;
            LS_sol_time(k,2,p)=toc;
            
            err_proj(k,p,2,m)=norm(data_vec-r1_data_vec*alpha_proj_F_LS);
            
            alpha_proj_F_LS_norm(k,p,m)=norm(alpha_proj_F_LS);
            
            %===============================================================================
            %%% vectorize + RFD
            siz_samp=size(r1_data_vec,1);
            tic
            R_rand_idx_vec=floor(siz_samp*rand(1,d_sec(p))); % random restriction with replacement
            randrest_vec_time(k,p)=toc;
            tic
            D_vec=2*(rand(1, siz_samp)<0.5)-1;
            randbit_vec_time(k,p)=toc;
            tic
            r1_data_vec_proj=fft(sparse(1:siz_samp, 1:siz_samp, D_vec)*[r1_data_vec, data_vec]);
            r1_data_vec_proj=r1_data_vec_proj(R_rand_idx_vec+1,:)/sqrt(d_sec(p));
            rand_proj_time(k,3,p)=toc;
            data_proj_vec=r1_data_vec_proj(:,end);
            r1_data_vec_proj(:,end)=[];
            tic
            alpha_proj_vec_LS=r1_data_vec_proj\data_proj_vec;
            LS_sol_time(k,3,p)=toc;
            
            err_proj(k,p,3,m)=norm(data_vec-r1_data_vec*alpha_proj_vec_LS);
            
            alpha_proj_vec_LS_norm(k,p,m)=norm(alpha_proj_vec_LS);
            
            t2=toc(t1);
            fprintf('Elapsed time: %f \n', t2)
            
        end
    end
            
end
end_time=toc(t_start);



rand_proj_1st_time_mean=mean(rand_proj_1st_time);
rand_proj_1st_time_mean=reshape(rand_proj_1st_time_mean, 2, length(dim_frac));
rand_proj_time_mean=mean(rand_proj_time);
rand_proj_time_mean=reshape(rand_proj_time_mean, 3, length(dim_frac));

LS_sol_1st_time_mean=mean(LS_sol_1st_time);
LS_sol_1st_time_mean=reshape(LS_sol_1st_time_mean, 2, length(dim_frac));
LS_sol_time_mean=mean(LS_sol_time);
LS_sol_time_mean=reshape(LS_sol_time_mean, 3, length(dim_frac));

time_1st_mean(1,:)=mean(sum(matgen_modewise_time,n_dims-1))+...
    rand_proj_1st_time_mean(1,:)+LS_sol_1st_time_mean(1,:); % 1st JL in Gaussian + RFD
time_1st_mean(2,:)=mean(mean(sum(randbit_modewise_time,n_dims-1)))+...
    mean(sum(randrest_modewise_time,n_dims-1))+mean(sum(matgen_F_modewise_time,n_dims-1))+...
    rand_proj_1st_time_mean(2,:)+LS_sol_1st_time_mean(2,:); % 1st JL in RFD + RFD

time_mean(1,:)=mean(sum(matgen_modewise_time,n_dims-1))+...
    mean(randbit_2nd_time)+rand_proj_time_mean(1,:)+mean(randrest_2nd_time)...
    +LS_sol_time_mean(1,:); % Gaussian + RFD
time_mean(2,:)=mean(mean(sum(randbit_modewise_time,n_dims-1)))+...
    mean(sum(randrest_modewise_time,n_dims-1))+mean(sum(matgen_F_modewise_time,n_dims-1))+...
    mean(randbit_2nd_time)+rand_proj_time_mean(2,:)+mean(randrest_2nd_time)...
    +LS_sol_time_mean(2,:); % RFD + RFD
time_mean(3,:)=mean(mean(randbit_vec_time))+rand_proj_time_mean(3,:)+mean(randrest_vec_time)...
    +LS_sol_time_mean(3,:); % vectorize + RFD


%%% save results
% save(sprintf('CPD_JL_Ntrials%d_r%d.mat', N_trials, r(idx)),...
%     'alpha_LS', 'alpha_proj_LS_norm', 'alpha_proj_F_LS_norm','alpha_proj_vec_LS_norm',...
%     'r', 'idx', 'dim_frac', 'dim_frac_sec', 'N_trials', 'end_time',...
%     'rand_proj_time','LS_sol_time','matgen_modewise_time','matgen_F_modewise_time','randbit_2nd_time','randrest_2nd_time',...
%     'rand_proj_time_mean','LS_sol_time_mean','randbit_modewise_time','randrest_modewise_time',...
%     'randbit_vec_time','randrest_vec_time','time_mean','d_sec',...
%     'err_true','err_proj','err_proj_1st','vec_target_dim',...
%     'time_1st_mean','rand_proj_1st_time','rand_proj_1st_time_mean',...
%     'LS_sol_1st_time','LS_sol_1st_time_mean','siz_samp')


%%% plot results
CPD_plots_new


