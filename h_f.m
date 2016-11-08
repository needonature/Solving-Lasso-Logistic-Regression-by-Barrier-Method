function result=h_f(beta,y,X)
    [n,p]=size(X);p=p-1;
    mu=exp(X*beta)./(1+exp(X*beta));
    result=X'*diag(mu.*(ones(n,1)-mu))*X;
end
