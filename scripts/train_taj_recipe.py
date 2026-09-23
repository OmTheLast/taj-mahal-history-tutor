"""Taj-only LoRA trial with fixed data order, validation selection and full recovery."""
import argparse,fcntl,hashlib,json,os,shutil,sys,time,platform
from pathlib import Path
from functools import partial
from importlib.metadata import version
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.tuner.utils import linear_to_lora_layers
from mlx_lm.tuner.trainer import grad_checkpoint
from mlx_lm.tuner.datasets import ChatDataset
from checkpoint_state import save_state,restore_state,atomic,sha
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data/taj_recipe_v1';RUN=ROOT/'runs/taj_recipe_v1';EXP=ROOT/'experiments/taj_recipe_v1'
def read(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def answer_loss(model,batch,lengths):
    logits=model(batch[:,:-1]);targets=batch[:,1:]
    positions=mx.arange(1,batch.shape[1])
    # Token indices stop strictly before the true length: padding is not a target.
    mask=(positions>=lengths[:,0:1])&(positions<lengths[:,1:2])
    ce=nn.losses.cross_entropy(logits,targets).astype(mx.float32)
    n=mask.sum();return (ce*mask).sum()/n,n
def batchify(examples,limit):
    width=32*((max(len(t) for t,_ in examples)+31)//32)
    assert width<=limit
    arr=np.zeros((len(examples),width),dtype=np.int32);lengths=[]
    for i,(tokens,offset) in enumerate(examples):
        assert 0<offset<len(tokens)<=limit
        arr[i,:len(tokens)]=tokens;lengths.append((offset,len(tokens)))
    return mx.array(arr),mx.array(lengths)
def validation(model,encoded,cfg):
    model.eval();weighted=0.;tokens=0
    for i in range(0,len(encoded),cfg['batch_size']):
        loss,n=answer_loss(model,*batchify(encoded[i:i+cfg['batch_size']],cfg['max_seq_length']))
        mx.eval(loss,n);weighted+=float(loss)*int(n);tokens+=int(n);mx.clear_cache()
    return weighted/tokens,tokens
def latest_complete(ck,settings):
    issues=[]
    for p in sorted(ck.glob('step*'),reverse=True):
        if p.name.endswith('.incomplete'):issues.append(str(p));continue
        try:
            meta=json.loads((p/'state.json').read_text())
            assert meta['settings']==settings,'Recovery settings changed'
            assert meta['step']==meta['schedule_position']
            assert sha(p/'adapters.safetensors')==meta['adapter_sha256']
            assert sha(p/'state.safetensors')==meta['state_sha256']
            return p,meta,issues
        except (OSError,ValueError,KeyError,AssertionError) as error:
            if 'settings changed' in str(error):raise
            issues.append(str(p))
    return None,None,issues
def repair_journal(path,step):
    if not path.exists():return
    raw=path.read_bytes();kept=[]
    for i,line in enumerate(raw.splitlines()):
        try:r=json.loads(line)
        except (ValueError,UnicodeDecodeError):
            assert i==len(raw.splitlines())-1,'Malformed journal before final line';continue
        if r['step']<=step:kept.append(json.dumps(r)+'\n')
    fixed=''.join(kept).encode()
    if fixed!=raw:
        path.with_name(f'{path.name}.before_recovery_{time.time_ns()}').write_bytes(raw)
        tmp=path.with_suffix('.tmp');tmp.write_bytes(fixed);tmp.replace(path)
def train(stop_after=None):
    RUN.mkdir(parents=True,exist_ok=True)
    lock=(RUN/'trainer.lock').open('a+')
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    config_path=ROOT/'configs/taj_recipe_v1.json';cfg=json.loads(config_path.read_text());manifest=json.loads((DATA/'manifest.json').read_text())
    assert manifest['status']=='frozen_for_training'
    assert sha(config_path)==manifest['config_sha256'] and sha(Path(__file__))==manifest['runner_sha256']
    assert sha(ROOT/'scripts/checkpoint_state.py')==manifest['recovery_helper_sha256']
    for filename in ['train.jsonl','valid.jsonl','schedule.json']:
        assert sha(DATA/filename)==manifest['hashes'][filename]
    settings={'config_sha256':sha(config_path),'data_manifest_sha256':sha(DATA/'manifest.json'),'runner_sha256':sha(Path(__file__)),'recovery_helper_sha256':sha(ROOT/'scripts/checkpoint_state.py'),'base_config_sha256':sha(ROOT/'models/qwen3-4b-base/config.json'),'base_index_sha256':sha(ROOT/'models/qwen3-4b-base/model.safetensors.index.json'),'software':{'python':sys.version,'platform':platform.platform(),'mlx':version('mlx'),'mlx_lm':version('mlx-lm')}}
    tests=json.loads((EXP/'runner_tests.json').read_text())
    assert tests['passed'] and tests['runner_sha256']==sha(Path(__file__))
    status_path=RUN/'status.json'
    if status_path.exists():
        old=json.loads(status_path.read_text());assert old['settings']==settings
        if old['status']=='completed':
            assert sha(RUN/'adapters/adapters.safetensors')==old['adapter_sha256'];print('Already completed');return
    ck=RUN/'checkpoints';ck.mkdir(exist_ok=True);adapter=RUN/'adapters';adapter.mkdir(exist_ok=True)
    atomic(adapter/'adapter_config.json',cfg)
    model,tok=load(str(ROOT/cfg['model']));mx.random.seed(cfg['seed']);np.random.seed(cfg['seed'])
    model.freeze();linear_to_lora_layers(model,cfg['num_layers'],cfg['lora_parameters'],use_dora=False)
    optimizer=optim.Adam(learning_rate=cfg['learning_rate']);grad_checkpoint(model.layers[0])
    trainrows=read(DATA/'train.jsonl');validrows=read(DATA/'valid.jsonl')
    ds=ChatDataset(trainrows,tok,mask_prompt=True);encoded=[ds.process(r) for r in trainrows]
    vd=ChatDataset(validrows,tok,mask_prompt=True);valid=[vd.process(r) for r in validrows]
    schedule=json.loads((DATA/'schedule.json').read_text());assert len(schedule)==cfg['iters']
    selected,before,issues=latest_complete(ck,settings)
    if issues:
        archive=RUN/f'recovery_quarantine_{time.time_ns()}';archive.mkdir()
        for item in issues:
            p=Path(item);p.rename(archive/p.name)
    step=0;seen=0;best=None
    if selected:
        restored=restore_state(selected,model,optimizer);step=restored['step'];seen=restored['trained_tokens'];best=restored['best_validation'];print('Restored',selected.name,'tokens',seen,flush=True)
    def checkpoint():
        save_state(ck/f'step{step:06d}',model,optimizer,{'step':step,'trained_tokens':seen,'schedule_position':step,'scheduler':{'type':'constant','learning_rate':cfg['learning_rate']},'best_validation':best,'settings':settings})
    if selected is None:checkpoint()
    log=RUN/'metrics.jsonl';repair_journal(log,step)
    def record(row):
        with log.open('a') as f:f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
    def status(kind,**extra):
        atomic(status_path,{'status':kind,'step':step,'total_steps':len(schedule),'trained_tokens':seen,'best_validation':best,'pid':os.getpid(),'settings':settings,'trainable_parameters':sum(v.size for _,v in tree_flatten(model.trainable_parameters())),**extra})
    status('running')
    history=read(log) if log.exists() else []
    if step==0 and not any(r.get('kind')=='validation' and r['step']==0 for r in history):
        loss,nt=validation(model,valid,cfg);record({'kind':'validation','step':0,'validation_loss':loss,'tokens':nt});print('Initial validation',loss,flush=True)
    model.train();lossgrad=nn.value_and_grad(model,answer_loss);state=[model.state,optimizer.state,mx.random.state]
    @partial(mx.compile,inputs=state,outputs=state)
    def update(batch,lengths):
        (loss,n),grads=lossgrad(model,batch,lengths);optimizer.update(model,grads);return loss,n
    tick=time.monotonic()
    for i in range(step,len(schedule)):
        batch=[encoded[j] for j in schedule[i]];loss,n=update(*batchify(batch,cfg['max_seq_length']))
        mx.eval(state,loss,n);step=i+1;seen+=int(n)
        record({'kind':'train','step':step,'train_loss':float(loss),'trained_tokens':seen,'elapsed_seconds_this_process':time.monotonic()-tick,'peak_memory_gb':mx.get_peak_memory()/1e9});mx.clear_cache()
        if step%25==0 or step==len(schedule):print('Step',step,'/',len(schedule),'loss',float(loss),'tokens',seen,flush=True)
        if step%cfg['steps_per_eval']==0 or step==len(schedule):
            loss,nt=validation(model,valid,cfg);record({'kind':'validation','step':step,'validation_loss':loss,'tokens':nt})
            if best is None or loss<best['loss']:best={'step':step,'loss':loss}
            model.train();print('Validation',step,loss,'best',best,flush=True)
        if step%cfg['save_every']==0 or step%cfg['steps_per_eval']==0 or step==len(schedule) or step==stop_after:
            checkpoint();status('running')
        if step==stop_after:status('paused_at_requested_checkpoint');print('Stopped after saved update',step,flush=True);return
    assert seen==manifest['scheduled_assistant_tokens'],(seen,manifest['scheduled_assistant_tokens'])
    assert best is not None and (ck/f'step{best["step"]:06d}').exists()
    chosen=ck/f'step{best["step"]:06d}'/'adapters.safetensors';shutil.copy2(chosen,adapter/'adapters.safetensors')
    status('completed',adapter_sha256=sha(adapter/'adapters.safetensors'),selected_checkpoint=f'step{best["step"]:06d}',selection='Lowest Taj validation loss among scheduled trained checkpoints; no benchmark selection.')
    print('TRAINING COMPLETE; selected',f'step{best["step"]:06d}',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stop-after',type=int);a=p.parse_args();train(a.stop_after)
