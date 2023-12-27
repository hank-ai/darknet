# Windows:  Extract CUDNN And Setup Paths

- download cudnn from https://developer.nvidia.com/rdp/cudnn-download 
- open powershell as admin. 

# Navigate to the directory where the cuDNN  is located (example)
    cd C:\Users\your_user\Downloads  (replace your_user)

# Unzip the cuDNN package
    Expand-Archive -Path "cudnn-windows-x86_64-*-archive.zip" -DestinationPath "cudnn_extracted" # this unzips the cudnn package

# Create the destination directories if they don't exist
    New-Item -ItemType Directory -Force -Path "C:\Program Files\NVIDIA\CUDNN\v8.x\bin"
    New-Item -ItemType Directory -Force -Path "C:\Program Files\NVIDIA\CUDNN\v8.x\include"
    New-Item -ItemType Directory -Force -Path "C:\Program Files\NVIDIA\CUDNN\v8.x\lib\x64"

# Copy the cuDNN files to the NVIDIA directory

# Copy DLL files
    Copy-Item -Path "cudnn_extracted\cudnn-windows-x86_64-8.7.0.84_cuda11-archive\bin\cudnn*.dll" -Destination "C:\Program Files\NVIDIA\CUDNN\v8.x\bin"

# Copy Header files
    Copy-Item -Path "cudnn_extracted\cudnn-windows-x86_64-8.7.0.84_cuda11-archive\include\cudnn*.h" -Destination "C:\Program Files\NVIDIA\CUDNN\v8.x\include"

# Copy Library files
    Copy-Item -Path "cudnn_extracted\cudnn-windows-x86_64-8.7.0.84_cuda11-archive\lib\x64\cudnn*.lib" -Destination "C:\Program Files\NVIDIA\CUDNN\v8.x\lib\x64"

# Remove the extracted folder to save space
    Remove-Item -Path "cudnn_extracted" -Recurse

#set the environment variable 
- Type Run and hit Enter.
- Issue the control run: `sysdm.cpl`
- Select the Advanced tab at the top of the window.
- Click Environment Variables at the bottom of the window.
- Add the NVIDIA cuDNN bin directory path to the PATH variable:
- under system varibles in "PATH"
  - Add: `C:\Program Files\NVIDIA\CUDNN\v8.x\bin`
  - Add: `C:\src\Darknet\build\src-cli\Release\darknet.exe`
  - Add: `C:\Program Files\darknet\bin\darknet.exe`

