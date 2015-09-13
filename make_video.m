% Increase for slower playback of video 
slo_mo = 3

% path to data
data_path = '~/Dropbox/Project 8883/';

% Load our tensor
al_data_blob = load(strcat(data_path, '800_1_pp1.mat'));
al_data = al_data_blob.phase_field_solid;
[m,n,k] = size(al_data);

% custom heatmap colors
color_map = [.675 .843 .125; 
             .886 .349 .133;
             .157 .22 .608;];

% Create Video Writer Object
filename = strcat(strcat('800_1_pp1_film_speed_', num2str(slo_mo)),'.avi');
writerObj = VideoWriter(filename);
open(writerObj);

% Each frame will be a figure
% hmap = HeatMap(al_data(:,:,1), 'Colormap', color_map);
colormap(color_map)
% axis tight
% set(gca,'nextplot','replacechildren');
% set(gcf,'Renderer','zbuffer');

for i = 1:k 
   imagesc(al_data(:,:,i));
   % Frame includes image data
   frame = getframe;
   % Adding the frame to the video object using the 'writeVideo' method
   for j = 1:slo_mo
       writeVideo(writerObj,frame);
   end
end

% Gotta close the video writer
close(writerObj);