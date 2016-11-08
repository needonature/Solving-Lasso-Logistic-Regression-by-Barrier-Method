function [f_diff,R]=Newton(R,y,X,t)
    [n,p]=size(X);p=p-1;
    beta=R(1:p+1);
    beta_=beta(2:p+1);
    c=R(p+2:end);
    alpha=0.2;
    beta0=0.9;
    f_star=306.476;

    p_pre=P(R,y,X,t);
    f_diff=[log(f(R,y,X)-f_star)];
    %first step
    t0=backtrack(R,y,X,t);
    v=-H(R,y,X,t)\g(R,y,X,t);
    R=R+t0*v;
    p_curr=P(R,y,X,t);
    f_diff=[f_diff,log(f(R,y,X)-f_star)];

    %following steps
    while(abs(p_curr-p_pre)>1e-9)
        t0=backtrack(R,y,X,t);
        v=-H(R,y,X,t)\g(R,y,X,t);
        R=R+t0*v;
        f_diff=[f_diff,log(f(R,y,X)-f_star)];
        p_pre=p_curr;
        p_curr=P(R,y,X,t);
    end
end
