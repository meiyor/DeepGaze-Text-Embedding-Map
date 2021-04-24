function p=stats_paired_tests(val1,val2)
[h1,p(1),stats1]=ttest(val1,val2);
[p(2),h2,stats2]=signtest(val1,val2);
[p(3),h3,stats3]=signrank(val1,val2);