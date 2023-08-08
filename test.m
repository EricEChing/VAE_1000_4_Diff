folder_path = 'C:\Users\ericc\PycharmProjects\VAE_10000\testMATfiles';  % Replace with the actual path to your folder
output_path = ''; % Replace with the actual path to your folder
% Use the dir function to get information about files in the folder
file_list = dir(fullfile(folder_path, '*.mat'));  % This will list all MATLAB files with extension '.m'
% Loop through the list of files and find those containing "000" in their filenames
files_with_00 = {};
for i = 1:length(file_list)
    file_name = file_list(i).name;
    if contains(file_name, '_00')
        files_with_00{end+1} = file_name;
    end
end

% Display the list of MATLAB files containing "000" in their filenames

for i = 1:length(files_with_00)
    old_file = files_with_00{i};
    old_file = fullfile(folder_path,old_file)
    old_T00 = load(old_file,"T00");
    old_T00_data = old_T00.T00;
    old_xymm = load(old_file,"xymm");
    old_xymm = old_xymm.xymm;
    old_zmm = load(old_file,"zmm");
    old_zmm = old_zmm.zmm;
    new_T00 = ctProc(old_T00_data,256,old_xymm,old_zmm);
    new_file_name = strcat("downsized_", old_file);
    save(fullfile(output_path,new_file_name),'new_T00')
end
