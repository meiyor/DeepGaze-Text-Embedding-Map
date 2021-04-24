function p=stats_normality(val)
[h1,p(1)]=jbtest(val,0.001);
[h2,p(2)]=kstest(val);
[h3,p(3)]=ttest(val);
[h4,p(4)]=swtest(val);