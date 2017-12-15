function [ W, AW, Anum, trounds ] = update_w(W, AW, Anum, Y, Bags, Beta, H, Alpha, S, rho, lambdaw, step_size0, trounds, k, disc, clipnorm, decayw)
   %%ADMM rule 1, update W via sgd, weight matrix in the model
   idx = 0;
   Wt = W;
   AWt = AW;
   Anumt = Anum;
   troundst = trounds;
   idx2 = 0;
   
   idxtmp = [1];
   idxtmp2 = [1];
   for i=1:length(Bags)
        bgsz = size(Bags{i},1);
        idx2 = idx2 + bgsz;
        idxtmp2 = [idxtmp2;idx2+1];
   end
   Yis = cell(length(Bags));
   Betais = Yis;
   His = Yis;
   Alphais = Yis;
   Cis = Yis;
   for i = 1:length(Bags)
       bgsz = size(Bags{i},1);
       Yi = Y(i,:);
       betai = Beta{i}';
       Hi = H(idx+1:idx+bgsz,:);
       Alphai = Alpha(idx+1:idx+bgsz,:);
       Ci = S(idx+1:idx+bgsz,:);
       Yis{i} = Yi;
       Betais{i} = betai;
       His{i} = Hi;
       Alphais{i} = Alphai;
       Cis{i} = Ci;
       idx = idx + bgsz;
       idxtmp = [idxtmp;idx+1];
   end
   ii = randperm(length(Bags));
   for i = 1:length(Bags)
       Bagi = Bags{ii(i)};
       Yi = Yis{ii(i)};
       betai = Betais{ii(i)};
       Hi = His{ii(i)};
       Alphai = Alphais{ii(i)};
       Ci = Cis{ii(i)};
       
       grad = GetGradWPerBag(W, Yi, Bagi, betai, Hi, Alphai, Ci, rho, lambdaw, k);
       grad_norm = norm(grad,'fro');
       if grad_norm > clipnorm
            grad = grad*(clipnorm/grad_norm);
       end
       
       step_size=step_size0/(1+decayw*trounds*step_size0);
       W = W*disc + (1-disc) * (W - step_size*grad);

       AW = [AW; {W}];
       Anum = Anum + 1;

       trounds = trounds + 1;
       
   end
   trounds = 0;
   AW(1:floor(length(AW)/2)) = [];
   Anum = floor(length(AW)/2);
end

function grad = GetGradWPerBag(W, Yi, Bagi, betai, Hi, Alphai, Ci, rho, lambdaw, k)
   Fi = get_g_func(Bagi*W);
   grad = lambdaw*W + rho*Bagi'*((Fi.*Ci-Hi-Alphai).*Ci.*Fi.*(1-Fi));
   grad(:,1:end-k) = grad(:,1:end-k) + Bagi'* ((betai'*(betai*Fi(:,1:end-k)-Yi)).*Fi(:,1:end-k).*(1-Fi(:,1:end-k)));
end