function t0=backtrack(R,y,X,t)
    [n,p]=size(X);p=p-1;
    beta=R(1:p+1);
    beta_=beta(2:p+1);
    alpha0=0.2;
    beta0=0.9;
    c=R(p+2:end);

    t0=1;
    v=-H(R,y,X,t)\g(R,y,X,t);
    while (P(R+t0*v,y,X,t)>P(R,y,X,t)+alpha0*t0*g(R,y,X,t)'*v)||sum(c+t0*v(p+2:end)<abs(beta_+t0*v(2:p+1)))~=0
        t0=beta0*t0;
    end
end
