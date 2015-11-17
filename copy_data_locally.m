
%data_path = 'C:\Users\almawa\Dropbox\Project 8883/';
data_path = '~/Dropbox/Project 8883/';
%new_path = 'D:\Spring 15\Github\project8883code\data\test/';
new_path = '~/Documents/repos/project8883code/data/test/';
listy = dir(data_path);
% disp(listy)
n = size(listy);
for i = 1:n
    if (size(listy(i).name,2) <= 4)
        continue;
    end
    if (strcmp(listy(i).name(end-3:end), '.mat'))
        temp = load(strcat(data_path, listy(i).name));
        phase_field_model = temp.phase_field_solid;
        new_filename = strcat(new_path, listy(i).name);
        save(new_filename, 'phase_field_model');
        
    end
end