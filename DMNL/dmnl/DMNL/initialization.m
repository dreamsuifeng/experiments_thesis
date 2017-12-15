function [ H, Beta, W, Alpha, S] = initialization(Y, Bags,param )
%initialization of variables

   k = param.k;

   
   insts = cell2mat(Bags);
   
   [idx, ~] = kmeans(insts, size(Y,2)+k);
   H  = ones(size(insts,1), size(Y,2)+k)*1e-3;
   for j=1:size(insts,1)
      H(j,idx(j)) = 1;
   end
   nmz = sum(H);
   for j=1:size(insts,1)
      H(j,idx(j)) = 1/sqrt(nmz(idx(j)));      
   end

   S = repmat(1./sqrt(nmz),size(H,1),1);
   
   [ Beta ] = update_beta(Bags, H, H./S, S );
   W = (rand(size(insts,2),size(Y,2)+k)*2-1)/1000;
   Alpha = zeros(size(H));
end

