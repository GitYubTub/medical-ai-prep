## To make new files and directories using PowerShell
- To create a folder in your current directory, use the command: "mkdir" + name of the directory you want to add
  - Note: if the directory's name has spaces in it, must wrap the name in quotes (EX: mkdir "My Machine Learning Project")
- To create an entire pathway with directories and sub-directories, type:  "mkdir" + whole path layout (EX: mkdir Project\Data\Raw_Images)
- To make a bunch of directories at once, make a list separated by spaces (EX: mkdir AP_Chem AP_Gov AP_Physics AP_Lang Multi_Calc)
  - Note: If any directory has spaces in its name, then must use quotes (EX: mkdir AP_Chem "AP US Gov" AP_Physics "AP English Lang")
- A PowerShell Range Loop can also be used to make a bunch of files with repeats in their names
<img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/0da03f38-ad0b-457f-b0ee-0318ea29c52c" />

- Stacking Pre-Made Sub-Branches can be used to make a directory with a bunch of subfolders (EX: mkdir "ML_Project\Data", "ML_Project\Models", "ML_Project\Notebooks", "ML_Project\Plots")
