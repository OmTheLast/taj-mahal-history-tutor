"""Evaluate the validation-selected adapter with the frozen original baseline protocol."""
import json,hashlib,os,time,datetime
from pathlib import Path
import mlx.core as mx
from mlx_lm import load,stream_generate
from mlx_lm.sample_utils import make_sampler
from run_taj_holdout import SYSTEM,user_prompt,sha,atomic,recover
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/taj_recipe_v1';BENCH=ROOT/'evaluation/taj_holdout/releases/v1'
def main():
    status=json.loads((RUN/'status.json').read_text());assert status['status']=='completed'
    manifest=json.loads((BENCH/'manifest.json').read_text())
    for file,key in [('questions.jsonl','questions_sha256'),('rubric.json','rubric_sha256')]:assert sha(BENCH/file)==manifest[key]
    assert sha(ROOT/'scripts/run_taj_holdout.py')==manifest['runner_sha256']
    adapter=RUN/'adapters';assert sha(adapter/'adapters.safetensors')==status['adapter_sha256']
    base=json.loads((BENCH/'runs/original/metadata.json').read_text());assert base['status']=='completed'
    rows=[json.loads(s) for s in (BENCH/'questions.jsonl').read_text().splitlines()]
    folder=RUN/'evaluation';folder.mkdir(exist_ok=True)
    settings={'model':'Qwen3-4B-Instruct-2507 + Taj recipe v1 LoRA','adapter_sha256':status['adapter_sha256'],'adapter_config_sha256':sha(adapter/'adapter_config.json'),'selected_checkpoint':status['selected_checkpoint'],'selection':status['selection'],'base_protocol_settings':base['settings'],'questions_sha256':manifest['questions_sha256'],'rubric_sha256':manifest['rubric_sha256'],'runner_sha256':sha(Path(__file__)),'system':SYSTEM,'temperature':0,'max_tokens':128,'seed':42}
    meta=folder/'metadata.json';old=json.loads(meta.read_text()) if meta.exists() else None
    if old:
        assert old['settings']==settings
        if old['status']=='completed':assert sha(folder/'responses.jsonl')==old['responses_sha256'];print('Already evaluated');return
    completed=recover(folder/'responses.jsonl');done={r['id'] for r in completed};assert len(done)==len(completed) and done<={q['id'] for q in rows}
    started=old['started_at'] if old else datetime.datetime.now(datetime.timezone.utc).isoformat()
    atomic(meta,{'status':'running','settings':settings,'started_at':started,'completed_items':len(done),'pid':os.getpid()})
    model,tok=load(str(ROOT/'models/qwen3-4b-base'),adapter_path=str(adapter));model.eval();mx.random.seed(42)
    with (folder/'responses.jsonl').open('a') as f:
        for q in rows:
            if q['id'] in done:continue
            prompt=tok.apply_chat_template([{'role':'system','content':SYSTEM},{'role':'user','content':user_prompt(q)}],tokenize=False,add_generation_prompt=True)
            tick=time.monotonic();pieces=list(stream_generate(model,tok,prompt=prompt,max_tokens=128,sampler=make_sampler(temp=0)))
            answer={'id':q['id'],'section':q['section'],'answer':''.join(p.text for p in pieces),'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'generation_tokens':pieces[-1].generation_tokens,'finish_reason':pieces[-1].finish_reason,'seconds':time.monotonic()-tick}
            f.write(json.dumps(answer,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno());done.add(q['id']);mx.clear_cache()
            atomic(meta,{'status':'running','settings':settings,'started_at':started,'completed_items':len(done),'pid':os.getpid()});print(q['id'],len(done),'/',len(rows),flush=True)
    reference={r['id']:r['prompt_sha256'] for r in map(json.loads,(BENCH/'runs/original/responses.jsonl').read_text().splitlines())}
    assert all(r['prompt_sha256']==reference[r['id']] for r in map(json.loads,(folder/'responses.jsonl').read_text().splitlines()))
    atomic(meta,{'status':'completed','settings':settings,'started_at':started,'finished_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'completed_items':len(done),'responses_sha256':sha(folder/'responses.jsonl'),'all_prompt_hashes_match_original':True,'peak_memory_gb':mx.get_peak_memory()/1e9})
if __name__=='__main__':main()
