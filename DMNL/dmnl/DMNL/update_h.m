function  [H] = update_h(H, F, S, A, Beta, Y, Bags,   insts,  lambda1 , rho, k,i, disc, decayh, squaredecayh, inner_iter )
     %%ADMM update rule 2, update H, the normalized clustering index matrix                   
    for i=1:inner_iter
       [pgrad,ngrad] = GetGradientH(H, F, S, A, Beta, Y, Bags, insts,  lambda1, rho, k, squaredecayh);
       H = H*(1-disc) + disc* (H.* (ngrad./pgrad).^(1/(1+sqrt(i)*decayh)));
        z=sum(H'*H);
        z = repmat(z,size(H,1),1);
        H = H./sqrt(z);
    end
end

function [pgrad,ngrad] = GetGradientH(H, F, S, A, Beta, Y, Bags, insts,  lambda1, rho, k,t)
   pgrad = zeros(size(H));
   ngrad = zeros(size(pgrad));
   idx = 0;
   for i=1:length(Bags)
       bgsz = size(Bags{i},1);
       Hi = H(idx+1:idx+bgsz,:);
       Ci = S(idx+1:idx+bgsz,:);
       Yi = Y(i,:);
       betai = Beta{i}';
       [pgradi,ngradi] = GetGradientHi(Hi,Yi,Ci,betai,k);
       pgrad(idx+1:idx+bgsz,:) = pgradi;
       ngrad(idx+1:idx+bgsz,:) = ngradi;
       idx = idx + bgsz;
   end
      
   pgrad0 = pgrad + rho*H + rho*(abs(A)+A)/2;
   ngrad0 = ngrad + rho*F.*S + rho*(abs(A)-A)/2 +  lambda1*insts*(insts'*H);
      
    pgrad = pgrad0 + H*ngrad0'*H*t;
    ngrad = ngrad0 + H*pgrad0'*H*t;
end

function [pgradi,ngradi] = GetGradientHi(Hi,Yi,Ci,betai,k)
   pgradi = zeros(size(Hi));
   ngradi = pgradi;
   Hitmp = Hi(:,1:end-k);
   Citmp = Ci(:,1:end-k);
   pgradi(:,1:end-k) = (betai'*(betai*(Hitmp./Citmp)))./Citmp;
   ngradi(:,1:end-k) = (betai'*Yi)./Citmp;
end

