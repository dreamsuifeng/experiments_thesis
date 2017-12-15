function G = get_g_func(Z)
%%Compute softmax function g for instance prediction
%%Given z=xw
   mx = max(Z,[],2);
   Z = Z - repmat(mx,1,size(Z,2));
   Z(Z<-15) = -15;
   P = exp(Z);
   s = sum(P,2);
   idx = s==0;
   invs = 1./s;
   invs(idx) = 0;
   G = diag(invs)*P;
end

