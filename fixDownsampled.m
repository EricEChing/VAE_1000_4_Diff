folder_path = 'C:\Users\ericc\PycharmProjects/VAE_10000/inputMATfiles';  % Replace with the actual path to your folder

% Use the dir function to get information about files in the folder
file_list = dir(fullfile(folder_path, '*.mat'));  % This will list all MATLAB files with extension '.m'
% Loop through the list of files and find those containing "000" in their filenames

for i = 1:length(file_list)
    file_name = file_list(i).name;
    load(file_name,'T00_new');
    new_T00 = T00_new;
    save(file_name,"new_T00");
end