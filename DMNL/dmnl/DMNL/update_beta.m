function [ Beta ] = update_beta(Bags, H, F, S)
   %%Update beta, the instance weights
   idx = 0;
   %H = (H + F.*S)/2;
   Beta = cell(length(Bags),1);
   for i=1:length(Bags)
       bgsz = size(Bags{i},1);
       Hi = H(idx+1:idx+bgsz,:);
       betai = zeros(bgsz,1);
       [~,lidx] = max(Hi,[],2);
       lp = unique(lidx);
       for j=1:length(lp)
           a = 1/sum(lidx==lp(j));
           betai(lidx==lp(j)) = a;           
       end
       Beta{i} = betai;
       idx = idx + bgsz;
   end
end

