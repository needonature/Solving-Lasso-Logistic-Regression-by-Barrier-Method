function result=g(R,y,X,t)
    lamda=15;
    [n,p]=size(X);p=p-1;
    beta=R(1:p+1);
    beta_=beta(2:p+1);
    c=R(p+2:end);
    mu=exp(X*beta)./(1+exp(X*beta));

    g_f=X'*(mu-y);
    g_fi_beta=[0;2*beta_./(c.^2-beta_.^2)];
    g_fi_c=[2*c./(beta_.^2-c.^2)];
    result=[t*g_f+g_fi_beta;t*lamda*ones(p,1)+g_fi_c];
end
