function [ W, AW, Anum, H, Beta] = dmnl_train(Y, Bags, param)
  %%training dmnl
   k = param.k;
   mx_rho = param.rho;
   lambdaw = param.lambdaw;
   step_size0 = param.step_size;
   lambda1 = param.lambda1;
   max_iter = param.max_iter;
   inner_iter = param.inner_iter;
   disc = param.disc;
   decayw = param.decayw;
   clipnorm = param.clipnorm;
   decayh = param.decayh;
   squaredecayh = param.decayhsquare;
   
   AW = 0;
   Anum = 0;
   trounds = 0;
   [ H, Beta, W, Alpha, S] = initialization(Y, Bags,param);
   insts = cell2mat(Bags);
   rho = 0;
   for i = 1:max_iter
       i
       %%update W
       for j=1:inner_iter
           [ W, AW, Anum, trounds ] = update_w(W, AW, Anum, Y, Bags, Beta, ...
                H, Alpha, S, rho, lambdaw, step_size0, trounds, k,disc, clipnorm, decayw);
       end
    
       Wm = 0;
       for j = 1:length(AW)
         Wm = Wm + AW{j};
       end
       Wm = Wm/length(AW);
       W = Wm;
       F = get_g_func(insts*W);


       %%update H
       [ H ] = update_h(H, F, S, Alpha, Beta, Y, Bags, insts,  lambda1, rho, k, i, disc, decayh, squaredecayh, inner_iter);

       %%update Beta
       [ Beta ] = update_beta(Bags, H, F, S);

       %%update Alpha
       [ Alpha ] = update_alpha( H, F, S, Alpha);
       
       %%update S
       [S] = update_s(H, F, S);
       
       
       if rho < mx_rho
            rho = rho + mx_rho/max_iter*2;
       else
            rho = mx_rho;
       end
       
%        [~, idx] = max(F,[],2);
%        [~, idx2] = max(y,[],2);
%        acc = sum(idx==idx2)/length(idx)
       
%        Fn = max(F(:,end-param.k+1:end),[],2);
%        nl = idx2>=20;
%        [~,~,~,AUC] = perfcurve(nl,Fn,1)
       
   end
end

