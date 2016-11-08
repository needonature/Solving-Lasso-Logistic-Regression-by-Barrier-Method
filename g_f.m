function result=g_f(beta,y,X)
    mu=exp(X*beta)./(1+exp(X*beta));
    result=X'*(mu-y);
end
