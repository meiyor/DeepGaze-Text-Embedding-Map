function Pnew=eval_stats(Data)
P(1,:)=stats_normality(Data.valscanpath_IG);
P(2,:)=stats_normality(Data.valscanpath_AUC);
P(3,:)=stats_normality(Data.valscanpath_NSS);
P(4,:)=stats_normality(Data.valscanpath_LL);
Pnew=bonf_holm(P);