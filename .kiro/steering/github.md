## inclusion: always

always use project's venv when running python

github access using:  
{  
"owner": "laniri",  
"repo": "drawings",  
"path": ".github/workflows"  
}

when using github MCP do the following when access fails:

1.  try again
2.  wait 30 seconds and try again
3.  ask the user how to proceed. don't look for work-arounds

when creating add hok files (guides, fixing scripts etcw) place them in tmp\_files folder that apear in .gitignore

if needed to perform aws operation use the following profile: d-9067931f77-921400262514-admin+Q

when debugging a failure. 
1. if log exists, use it before deciding about the root cause
2. read last 2 commits comments of the files that are suspicious to see if there is relvant info there

when adding a new feature read last 2 commits comments of the files that are about to be modified

when inspecting logs and reports, pay attention to time zone diffrences that can happen between local, aws and github
