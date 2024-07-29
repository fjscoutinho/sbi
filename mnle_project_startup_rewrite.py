import os
import site
import io

append_folders_string = r'''import sys

if sys.platform.startswith('win32'):
    ### FOLDER SETTINGS ###
    #PROJECT_DIR = r'C:\Users\Filipe\Documents\GitHub'
    PROJECT_DIR = r'E:\GitHub'
    sbi_folder = PROJECT_DIR + r'\sbi'
    mnle_project_folder = PROJECT_DIR + r'\mnle_project'
    ddm_task_folder = PROJECT_DIR + r'\sbibm-ddm-task'
    mnle_for_ddms_folder = PROJECT_DIR + r'\mnle-for-ddms'
    nsf_folder = PROJECT_DIR + r'\lfi\src\nsf'
    SAVE_DIR = r'E:\ProjectDumps\mnle_project'
    tests_folder = SAVE_DIR + r'\tests'
    trainers_folder = tests_folder + r'\\trainers\\'
    data_folder = SAVE_DIR + r'\data'
    
elif sys.platform.startswith('linux'):
    ### FOLDER SETTINGS ###
    PROJECT_DIR = r'/home/filipe/Documents/GitHub'
    sbi_folder = PROJECT_DIR + r'/sbi'
    mnle_project_folder = PROJECT_DIR + r'/mnle_project'
    ddm_task_folder = PROJECT_DIR + r'/sbibm'
    mnle_for_ddms_folder = PROJECT_DIR + r'/mnle-for-ddms'
    nsf_folder = PROJECT_DIR + r'/lfi/src/nsf'
    SAVE_DIR = r'/home/filipe/Documents/ProjectDumps/mnle_project'   
    tests_folder = SAVE_DIR + r'/tests'
    trainers_folder = tests_folder + r'/trainers/'
    data_folder = SAVE_DIR + r'/data'

project_folders = [sbi_folder, mnle_project_folder, ddm_task_folder, mnle_for_ddms_folder, nsf_folder, tests_folder, trainers_folder, data_folder]
for project_folder in project_folders:
    if project_folder not in sys.path:
        sys.path.append(project_folder)
'''

# Grab directory where the user site packages can be found
user_site_dir = site.getusersitepackages()
#user_site_dir = r'/home/filipe/.local/lib/python3.11/site-packages/'

# Define path of usercustomize.py file which is going to contain our target paths
filename = 'usercustomize.py'
user_customize_filename = os.path.join(user_site_dir, filename)

# Create the user site directory if it does not already exist
if os.path.exists(user_site_dir):
    print("User site dir already exists")
else:
    print("Creating site dir")
    os.makedirs(user_site_dir)

# Create usercustomize.py if it does not already exist
if not os.path.exists(user_customize_filename):
    print("Creating {filename}".format(filename=user_customize_filename))
    file_mode = 'w+t'
    # Creating a file at specified location
    with open(os.path.join(user_site_dir, filename), 'w') as f:
        pass
        # To write data to new file uncomment
        # this f.write("New file created")
else:
    print("{filename} already exists".format(filename=user_customize_filename))
    file_mode = 'r+t'

# Fill in usercustomize.py so that necessary folders are appended to sys.path as desired
with io.open(user_customize_filename, file_mode) as user_customize_file:
    existing_text = user_customize_file.read()

    # delete any existing content
    user_customize_file.truncate(0)
    user_customize_file.seek(0)

    # file pointer should already be at the end of the file after read()
    user_customize_file.write(append_folders_string)

    print("Code to append folders added to {filename}".format(filename=user_customize_filename))