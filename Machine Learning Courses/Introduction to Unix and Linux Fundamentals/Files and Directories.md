## Files and Directories

### Unix File Management
Ordinary (Regular) Files: Every day files. Contain text, data, or program instructions
- the leaves at the very end of the branches
- Simply hold data or instructions
  
Directories: Similar to folders on Mac or Windows; used to bundle similar files together
- the branches that hold the leaves
- Only the Kernel can write to a directory file. When you create or delete a folder, you are asking the OS Kernel to update its master map of the system tree
  
Special Files: act as pointers (shortcuts) to hardware
- If a program writes data to a special file associated with a printer, that data is instantly sent to the physical printer
- If it reads from a special file linked to a keyboard, it captures your keystrokes

### Listing Files
The ls command is basically used for viewing all list of files/directories. Ls command looks like:
<img width="493" height="138" alt="image" src="https://github.com/user-attachments/assets/f37c447e-5099-409c-a387-8a7513e7bc17" />

### ls Command
Command + Option/Flag + Argument
ls is a command, and -l is one of its options
- ls with -l option gives more information about each of the files listed in a specific directory
- For PowerShell, just need ls (PowerShell assumes -l is -LiteralPath, which needs a path folder string after it)
  
From the image above, this is what each column means:
- Column 1: Mode (The File Attributes)
  - Position 1: If the very first letter is a d, the item is a folder (a branch). If it is a dash -, it is an ordinary file (a leaf).
  - Position 2: If an "a" is there, it tells the user that the files are stored independently of their primary computer, and the file remains entirely safe if anything happens to this computer. (the files are in OneDrive, therefore, a is in the second position)
  - Position 3: If a "r" is there, it means the file is locked. You can open it and look at it, but Windows will block you from editing or saving changes to it.
  - Position 4: If a "h" is there, it means the file is normally invisible in your standard Windows File Explorer, though it still shows up when you force a list command in the terminal.
  - Position 5: If a "s" is there, it means the file is a critical file that the Windows operating system needs to run properly.
    
- Column 2: LastWriteTime (The Timestamp): This column shows the exact date and time the folder or file was last written to or altered.
 
- Column 3: Length (The File Size)
  - This column tells you how much space the item occupies on your storage drive, measured in bytes
  - Will not display for folders due to processing speed
 
- Column 4: Name (The Identity)
  - This is the literal name of the file or directory, including its file extension if it has one.
  - Folders will not have file extensions
 
### LS Metacharacters
In the Is listing example, every file line began with a d, -, or l. These characters indicate the type of file that's listed
Only need d, -, and l for PowerShell
- $-$ (Regular File): Your basic scripts (.py), data files (.csv), and text documents (.txt).
- d (Directory): Folders used to structure the system tree.
- l (Symbolic Link): A shortcut pointing directly to another file.
  
Metacharacters (Wildcards)
These are special symbols that the shell intercepts and interprets as special commands made by the user rather than literal text.
The two most common are the Asterisk (*) and the Question Mark (?).
- The Asterisk (*): Matches Zero or More Characters
  - Think of the asterisk as an "any fill-in-the-blank" card.
  <img width="765" height="447" alt="image" src="https://github.com/user-attachments/assets/0feacc27-e864-4f1d-a5f7-05d4c9d16c24" />
  
- The Question mark (?): it represents exactly one character slot.
  - Used to find matches with a single character.
  <img width="674" height="430" alt="image" src="https://github.com/user-attachments/assets/1e456496-2f0a-4896-bb4b-022d95f1d373" />

### Hidden Files
- Hidden files will contain a "h" in position 4 of their mode column
- They will not show when a normal ls command is run
- revealed with "ls -force"

<img width="500" height="250" alt="image" src="https://github.com/user-attachments/assets/d5f34051-9607-4be4-a965-4e188fe45071" /> <img width="500" height="250" alt="image" src="https://github.com/user-attachments/assets/81164024-b808-43b0-a046-d482fba44f20" />

Directories:
- Single Dot (.): Represents the current directory
- Double Dot (..): Represents the parent directory (the folder directly above in the file system tree)
  - Can change to the parent directory by typing "cd .."
  - will work until there are no more directories above

### Other LS Command Variants
- Single Column Layout: output only the names as a bare, clean list of strings without any table headers or metadata columns
  - "ls -Name"
- Directory Information Only: just the structural folder nodes and skip all the individual files entirely
  - "ls -Directory"
- Sorting by Time: the most recently edited file appears at the very bottom line of the terminal window
  - "ls | Sort-Object LastWriteTime"

### Displaying Files Recursively
In terminal file management, a recursive operation means that a command will not just look at the surface level of the directory you are standing in. It will dive deep into every subfolder, open those subfolders up, list everything inside them, and keep diving down until it has mapped out the entire underlying object-oriented map
- The Direct Parameter
  - "ls -Recurse" will flood your terminal screen with a complete, structured view of your current folder alongside every hidden sub-leaf inside it
- The Deep-Filter Search
  - ls -Recurse *something" PowerShell will instantly dig through every single branch of your directory tree to fish out that exact file path for you

## To make new files and directories using PowerShell
- To create a folder in your current directory, use the command: "mkdir" + name of the directory you want to add
  - Note: if the directory's name has spaces in it, must wrap the name in quotes (EX: mkdir "My Machine Learning Project")
- To create an entire pathway with directories and sub-directories, type:  "mkdir" + whole path layout (EX: mkdir Project\Data\Raw_Images)
- To make a bunch of directories at once, make a list separated by spaces (EX: mkdir AP_Chem AP_Gov AP_Physics AP_Lang Multi_Calc)
  - Note: If any directory has spaces in its name, then must use quotes (EX: mkdir AP_Chem "AP US Gov" AP_Physics "AP English Lang")
- A PowerShell Range Loop can also be used to make a bunch of files with repeats in their names
<img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/0da03f38-ad0b-457f-b0ee-0318ea29c52c" />

- Stacking Pre-Made Sub-Branches can be used to make a directory with a bunch of subfolders (EX: mkdir "ML_Project\Data", "ML_Project\Models", "ML_Project\Notebooks", "ML_Project\Plots")

## Side Note (LiteralPath)
LiteralPath specifies the exact location of a file or folder.
The difference between LiteralPath and Path is how the terminal handles wildcard characters (like *, ?, or []).
- The Standard Path Parameter allows wildcards
- The LiteralPath Parameter does not allow wildcards

The Path Parameter (Allows Wildcards)
- If you type ls *.csv, it looks for anything ending in .csv.
- If you type ls photo[1-3].png, it looks for photo1.png, photo2.png, and photo3.png.

The LiteralPath Parameter (No Wildcards Allowed)
The LiteralPath parameter tells PowerShell, "Do not try to be smart. Do not interpret any character as a search filter. Take the folder name exactly as I typed it, word-for-word."
  
Important when there are wildcard characters in the file name EX: "AI_Med_[2026]"








