"""First-party Azure CLI control, restricted to this experiment's existing VM.

Authentication is a separate, user-confirmed step. No new Azure roles or
service principals are created. Credentials stay in ignored vendor/private.
"""
from pathlib import Path
import argparse,json,os,subprocess,time

ROOT=Path(__file__).resolve().parents[1]
SUBSCRIPTION='1d3d1d9e-c929-488c-9a57-d2d43cb71f0b'
GROUP='rg-aiexp-19'
VM='aiexp-19'
TENANT='univtokyo.onmicrosoft.com'


def environment():
    directory=ROOT/'vendor/private/azure_cli';directory.mkdir(parents=True,exist_ok=True);directory.chmod(0o700)
    return {**os.environ,'AZURE_CONFIG_DIR':str(directory),'AZURE_CORE_COLLECT_TELEMETRY':'0','AZURE_CORE_NO_COLOR':'1'}


def cli(*args):
    binary=ROOT/'.venv-azure/bin/az'
    if not binary.exists():raise FileNotFoundError('Install the isolated official Azure CLI first')
    result=subprocess.run([str(binary),*args,'--only-show-errors'],env=environment(),capture_output=True,text=True)
    if result.returncode:raise RuntimeError(result.stderr.strip() or 'Azure CLI command failed')
    return result.stdout.strip()


def power_state():
    return cli('vm','get-instance-view','--subscription',SUBSCRIPTION,'--resource-group',GROUP,'--name',VM,
        '--query',"instanceView.statuses[?starts_with(code, 'PowerState/')].code | [0]",'-o','tsv')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['assert-ready','status','deallocate']);a=parser.parse_args()
    # Read the account without printing personal account information or tokens.
    account=json.loads(cli('account','show','--subscription',SUBSCRIPTION,'-o','json'))
    if account.get('id','').lower()!=SUBSCRIPTION:raise ValueError('Unexpected Azure subscription')
    if a.command=='deallocate':
        cli('vm','deallocate','--subscription',SUBSCRIPTION,'--resource-group',GROUP,'--name',VM)
        for _ in range(12):
            state=power_state()
            if state=='PowerState/deallocated':break
            time.sleep(5)
        else:raise RuntimeError('Deallocation was requested but not yet verified')
    else:state=power_state()
    print(json.dumps({'vm':VM,'power_state':state,'deallocated':state=='PowerState/deallocated'}),flush=True)


if __name__=='__main__':main()
