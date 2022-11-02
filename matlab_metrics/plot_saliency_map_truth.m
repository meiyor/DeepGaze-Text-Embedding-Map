function plot_saliency_map_truth(path,name,vals)
close all;
%% fixation dataset with all the val values
preffix='/home/jmm_vivobook_asus/DeepGaze_project/COCO/COCO_subfolder_output/fixations/val/';
%% images subfolder selected when you desire and in your local computer. These are local directions of my previous val folder but you can locate them as you prefer
preffix_img='/home/jmm_vivobook_asus/DeepGaze_project/COCO/COCO_subfolder_output/val/';
val_suf=strsplit(path,'.');
IMor=imread([preffix_img val_suf{1} '.jpg']);
Data=load([preffix path]);
fix_vals_x=[];
fix_vals_y=[];
for i=1:length(Data.gaze)
    fix{i}=Data.gaze(i).fixations;
    if length(fix{i})~=0
        fix_vals_x=[fix_vals_x fix{i}(:,1)'];
        fix_vals_y=[fix_vals_y fix{i}(:,2)'];
    end;
end;
hh1 = hist3([fix_vals_x',fix_vals_y'],'Nbins',[vals vals+5]);
hh2 = hist3([fix_vals_y',fix_vals_x'],'Nbins',[vals vals+5]);
%hh3=hh1;
hh3=(hh1+hh2)/2;
hh3=(hh3-min(hh3))./(max(hh3)-min(hh3));
figure;
imshow(IMor);
hold on;
imagesc(imresize(hh3,[480,640]),'AlphaData',.4);
colormap(jet);
caxis([0.4 1.0])
colorbar;
saveas(gcf,['groundtruth_' name '.fig'])
A=1;
