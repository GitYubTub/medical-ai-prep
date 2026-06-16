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
ls with -l option gives more information about each of the files listed in a specific directory
For PowerShell just need ls (Powershell assumes -l is -LiteralPath, which needs a pat folder string after it)
