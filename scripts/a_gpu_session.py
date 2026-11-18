"""Run an explicit experiment command; always back up then deallocate this VM.

Use only after confirmed Azure CLI sign-in. The Mac must remain awake and
connected. Ending model interaction does not end this local cleanup process.
"""
from pathlib import Path
import argparse,subprocess,sys,os,json,signal


def main():
    p=argparse.ArgumentParser();p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    if a.command and a.command[0]=='--':a.command=a.command[1:]
    if not a.command:p.error('Supply an explicit workload command')
    root=Path(__file__).resolve().parents[1];os.chdir(root)
    subprocess.run([sys.executable,'scripts/a_azure.py','assert-ready'],check=True)
    workload=None;backup=False;stopped=False;failure=None;stop_failure=None;workload_failure=None
    def interrupted(signum,frame):raise KeyboardInterrupt('Interrupted; backing up and deallocating')
    signal.signal(signal.SIGTERM,interrupted)
    try:
        workload=subprocess.run(a.command).returncode
    except BaseException as error:
        workload_failure=type(error).__name__
        raise
    finally:
        try:
            subprocess.run(['bash','scripts/a_backup.sh'],env={**os.environ,'BAND2TAB_BACKUP_DATA':'1'},check=True)
            subprocess.run([sys.executable,'scripts/a_verify_backup.py'],check=True);backup=True
        except Exception as error:
            failure=type(error).__name__
        finally:
            # Once cleanup starts, repeated Ctrl-C must not skip deallocation.
            signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGINT,signal.SIG_IGN)
            try:
                subprocess.run([sys.executable,'scripts/a_azure.py','deallocate'],check=True);stopped=True
            except Exception as error:stop_failure=type(error).__name__
            finally:
                path=Path('reports/azure_session_cleanup.json');path.parent.mkdir(exist_ok=True)
                path.write_text(json.dumps({'workload_exit_code':workload,'workload_failure':workload_failure,'backup_checkpoints_verified':backup,'deallocated':stopped,'backup_failure':failure,'deallocation_failure':stop_failure},indent=2))
    if workload or not backup or not stopped:raise SystemExit(workload or 1)


if __name__=='__main__':main()
