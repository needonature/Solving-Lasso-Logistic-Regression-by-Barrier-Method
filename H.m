function result=H(R,y,X,t)
    [n,p]=size(X);p=p-1;
    beta=R(1:p+1);
    beta_=beta(2:p+1);
    c=R(p+2:end);
    mu=exp(X*beta)./(1+exp(X*beta));

    h_f=X'*diag(mu.*(ones(n,1)-mu))*X;
    B1=diag(1./(c+beta_).^2+1./(c-beta_).^2);
    B2=diag(1./(c+beta_).^2-1./(c-beta_).^2);
    B11=[zeros(1,1+p+p);[zeros(p,1),B1,B2]];
    B22=[zeros(p,1),B2,B1];
    result=[[t*h_f,zeros(1+p,p)]+B11;B22];
end
