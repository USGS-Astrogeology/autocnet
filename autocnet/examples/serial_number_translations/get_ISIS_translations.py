import glob
import os

files = glob.glob('/usgs/cpkgs/isis3/data/*/translations/*SerialNumber????.trn')
files.sort(reverse=True)

for i,file in enumerate(files):

    if file[:-8] != files[i-1][:-8]:

        # the second to last value in the directory path is the mission dir
        mission_dir  = os.path.dirname(file).split('/')[-2]
        file_name = os.path.basename(file)
        base, ext = os.path.splitext(file_name)
        # remove the last 4 characters (i.e. version numbers) from base name
        new_name = mission_dir + base[0].upper() + base[1:-4] + ext
    
        with open(file, 'r') as f:
            file_lines = f.readlines()
            while "  Auto\n" in file_lines:
                file_lines.remove("  Auto\n")

        print("COPYING [", file, "]\nTO [", new_name, "]\n")

        with open(new_name,  'w') as f:
            for t in file_lines:
                f.write(t)
