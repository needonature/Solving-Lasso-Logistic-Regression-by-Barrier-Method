function result=f(R,y,X)
    lamda=15;
    [n,p]=size(X);p=p-1;
    beta=R(1:p+1);
    c=R(p+2:end);
    result=-sum(y.*(X*beta))+sum(log(1+exp(X*beta)))+lamda*sum(c);
end
