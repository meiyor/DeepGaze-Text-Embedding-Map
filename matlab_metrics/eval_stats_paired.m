function Pnew=eval_stats_paired(Data)
P(1,:)=stats_paired_tests(Data.valbaseline_IG,Data.valscanpath_IG);
P(2,:)=stats_paired_tests(Data.valbaseline_AUC,Data.valscanpath_AUC);
P(3,:)=stats_paired_tests(Data.valbaseline_NSS,Data.valscanpath_NSS);
P(4,:)=stats_paired_tests(Data.valbaseline_LL,Data.valscanpath_LL);
%% bonferroni holm correction for each method evaluation
Pnew=bonf_holm(P);
