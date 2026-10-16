@echo off
rem FlyBot on AWS: one-command deploy for Windows.
rem   deploy.bat                 full deploy (idempotent, re-run after any change)
rem   deploy.bat --dry-run       checks and plan only, changes nothing
rem   deploy.bat --show-ip       Elastic IP and DNS record of the existing stack
rem   deploy.bat --only <step>   one step; see deploy.bat --help
rem Runs deploy.sh with Git for Windows' bash (not the WSL bash in System32).
setlocal EnableExtensions
cd /d "%~dp0"

set "BASH_EXE="
if defined GIT_BASH if exist "%GIT_BASH%" set "BASH_EXE=%GIT_BASH%"
if not defined BASH_EXE if exist "%ProgramFiles%\Git\bin\bash.exe" set "BASH_EXE=%ProgramFiles%\Git\bin\bash.exe"
if not defined BASH_EXE if exist "%ProgramFiles(x86)%\Git\bin\bash.exe" set "BASH_EXE=%ProgramFiles(x86)%\Git\bin\bash.exe"
if not defined BASH_EXE if exist "%LocalAppData%\Programs\Git\bin\bash.exe" set "BASH_EXE=%LocalAppData%\Programs\Git\bin\bash.exe"
if not defined BASH_EXE (
  for /f "delims=" %%G in ('where git.exe 2^>nul') do (
    if not defined BASH_EXE if exist "%%~dpG..\bin\bash.exe" set "BASH_EXE=%%~dpG..\bin\bash.exe"
  )
)
if not defined BASH_EXE (
  echo Git for Windows was not found. Install it from https://git-scm.com/download/win
  echo or set GIT_BASH to the full path of Git's bin\bash.exe, then run deploy.bat again.
  exit /b 1
)

rem Help needs no settings file.
set "WANTS_HELP="
for %%A in (%*) do (
  if /i "%%~A"=="--help" set "WANTS_HELP=1"
  if /i "%%~A"=="-h" set "WANTS_HELP=1"
)

if not defined WANTS_HELP if not exist "deploy.env" (
  copy /y "deploy.env.example" "deploy.env" >nul
  echo deploy.env was created from deploy.env.example.
  echo Fill it in ^(domain, emails, EZVIZ keys, Telegram, SMTP, Google service account^), save it,
  echo then run:  aws login --profile sentinel
  echo and run deploy.bat again.
  start "" notepad "deploy.env"
  exit /b 1
)

"%BASH_EXE%" "%~dp0deploy.sh" %*
set "RC=%ERRORLEVEL%"
if "%RC%"=="2" (
  echo.
  echo The AWS session has expired. Run:  aws login --profile sentinel
  echo then run deploy.bat again.
) else if not "%RC%"=="0" (
  echo.
  echo deploy.sh stopped with exit code %RC%. Details: infra\.build\deploy-dev.log
)
exit /b %RC%
