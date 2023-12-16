@echo off

if not exist build (
	echo You need to manually run through the build steps at least once before you use this script.
	echo Please see https://github.com/hank-ai/darknet#building for details.
	exit /b 1
)

echo on
cls

cd build

set VCPKG_PATH=C:/src/vcpkg
set ARCHITECTURE=x64

rem Pick just 1 of the following -- either Release or Debug
set BUILD_TYPE=Release
rem set BUILD_TYPE=Debug

cmake -DCMAKE_BUILD_TYPE=%BUILD_TYPE% -DCMAKE_TOOLCHAIN_FILE=%VCPKG_PATH%/scripts/buildsystems/vcpkg.cmake ..
if %ERRORLEVEL% neq 0 goto END

msbuild.exe /property:Platform=%ARCHITECTURE%;Configuration=%BUILD_TYPE% /target:Build -maxCpuCount -verbosity:normal -detailedSummary darknet.sln
if %ERRORLEVEL% neq 0 goto END

msbuild.exe /property:Platform=%ARCHITECTURE%;Configuration=%BUILD_TYPE% PACKAGE.vcxproj
if %ERRORLEVEL% neq 0 goto END

echo Done!
echo Make sure you run the Darknet installation wizard to install Darknet:
dir *.exe

:END
cd ..
