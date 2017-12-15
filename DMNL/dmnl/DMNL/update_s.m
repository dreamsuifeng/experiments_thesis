function S = update_s(H, F, S)
    %update S, normalization coefficient of H
       %H = (H + F.*S)/2;
       [~,idx] = max(H,[],2);     
       Hhat  = ones(size(H,1), size(H,2))*1e-3;
       for j=1:size(H,1)
          Hhat(j,idx(j)) = 1;
       end
       nmz = sum(Hhat+1e-6);
       S = repmat(1./sqrt(nmz),size(Hhat,1),1);
end