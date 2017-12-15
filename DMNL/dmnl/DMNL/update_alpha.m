function [ alpha ] = update_alpha( H, F, S, alpha)
  %%ADMM update rule 3, update alpha
  alpha = alpha + H-F.*S;
end

