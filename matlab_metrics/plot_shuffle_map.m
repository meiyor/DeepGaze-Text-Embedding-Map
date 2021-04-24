function plot_shuffle_map(path,name)
close all
preffix='/home/jmm_vivobook_asus/DeepGaze_project/COCO/COCO_subfolder_output/val/';
IMoriginal=imread([preffix path]);
IMTEM=dlmread(['/home/jmm_vivobook_asus/DeepGaze_project/deepgaze_master_Evaluation/results_shuffle_TEM_scan/val_folder/' path '_results_saliency.txt']);
figure();
imshow(IMoriginal);
hold on;
imagesc((IMTEM-min(IMTEM))./(max(IMTEM)-min(IMTEM)),'AlphaData', .4);
caxis([0.8 1.0])
colormap('jet');
colorbar
saveas(gcf,['TEM_shuffle' name '.fig'])