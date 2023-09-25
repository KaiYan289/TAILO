import argparse
import subprocess
    
def get_git_diff():
    tmp = subprocess.run(['git', 'diff', '--exit-code'], capture_output=True)
    tmp2 = subprocess.run(['git', 'diff', '--cached', '--exit-code'], capture_output=True)
    return tmp.stdout.decode('ascii').strip() + tmp2.stdout.decode('ascii').strip()
    
def git_commit(runtime):
    tmp = subprocess.run(['git', 'commit', '-a', '-m', runtime], capture_output=True)
    return tmp.stdout.decode('ascii').strip()