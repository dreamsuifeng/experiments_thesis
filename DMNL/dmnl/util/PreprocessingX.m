function [X_out]=PreprocessingX(X,kernel,kernel_method,scale)
w = warning ('on','all');
rmpath('folderthatisnotonpath');
warning(w);
id = w.identifier;
warning('off',id);
rmpath('folderthatisnotonpath');

if(strcmp(kernel,'kernel')==1)
    [X_out]=kernelize(X,kernel_method,scale);
end
end