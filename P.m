function result=P(R,y,X,t)
    [n,p]=size(X);p=p-1;
    lamda=15;
    beta=R(1:1+p);
    c=R(p+2:end);
    f=@(beta,y,X) -y'*X*beta+sum(log(1+exp(X*beta)))+lamda*sum(c);
    result=t*f(beta,y,X)-sum(log(c.^2-beta(2:end).^2));
end
