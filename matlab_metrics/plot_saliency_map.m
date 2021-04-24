function plot_saliency_map(path,name)
close all
preffix='/home/jmm_vivobook_asus/DeepGaze_project/COCO/COCO_subfolder_output/val/';
IMoriginal=imread([preffix path]);
IMbase=dlmread(['/home/jmm_vivobook_asus/DeepGaze_project/deepgaze_master_Evaluation/results_baseline_real_fixations/val_folder/' path '_results_saliency.txt']);
IMTEM=dlmread(['/home/jmm_vivobook_asus/DeepGaze_project/deepgaze_master_Evaluation/def_approach_including_scanpath/val_folder/' path '_results_saliency.txt']);
figure();
imshow(IMoriginal);
hold on;
imagesc((IMbase-min(IMbase))./(max(IMbase)-min(IMbase)),'AlphaData', .4);
caxis([0.8 1.0])
colormap('jet');
colorbar
saveas(gcf,['baseline_' name '.fig'])
close all
figure();
imshow(IMoriginal);
hold on;
imagesc((IMTEM-min(IMTEM))./(max(IMTEM)-min(IMTEM)),'AlphaData', .4);
caxis([0.8 1.0])
colormap('jet');
colorbar
saveas(gcf,['TEM_' name '.fig'])