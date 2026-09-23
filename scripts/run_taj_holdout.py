"""Original-model baseline for a frozen audited Taj release; no adapter loading."""
import argparse,json,hashlib,os,time,datetime,sys,platform
from importlib.metadata import version
from pathlib import Path
import mlx.core as mx
from mlx_lm import load,stream_generate
from mlx_lm.sample_utils import make_sampler
ROOT=Path(__file__).resolve().parents[1]
SYSTEM='You are a history tutor. Answer each question directly using the requested format. Give accurate information and do not invent facts.'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def atomic(p,obj):
 tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2))
 with tmp.open('rb')as f:os.fsync(f.fileno())
 tmp.replace(p)
def recover(p):
 if not p.exists():return []
 raw=p.read_bytes();rows=[];offset=0;parts=raw.splitlines(keepends=True)
 for i,line in enumerate(parts):
  try:row=json.loads(line)
  except (ValueError,UnicodeDecodeError):
   assert i==len(parts)-1,'Malformed row before file end'
   p.with_name(p.name+f'.interrupted-{time.time_ns()}').write_bytes(raw[offset:])
   with p.open('r+b')as f:f.truncate(offset)
   break
  rows.append(row);offset+=len(line)
 if rows and p.read_bytes()and not p.read_bytes().endswith(b'\n'):
  with p.open('ab')as f:f.write(b'\n');f.flush();os.fsync(f.fileno())
 return rows
def user_prompt(q):
 if q['section']=='taj_mcq':return q['question']+'\n'+'\n'.join(f'{k}) {q["options"][k]}'for k in 'ABCD')+'\nAnswer with exactly one letter: A, B, C or D.'
 if q['section']=='evidence_abstention':return q['question']+'\nUse only this note. If the requested information is absent, explicitly say it is not provided. Answer in one sentence.'
 return q['question']+'\nGive only the requested factual details in one sentence.'
def main(release):
 folder=(ROOT/'evaluation/taj_holdout/releases'/release).resolve();assert folder.parent==ROOT/'evaluation/taj_holdout/releases';mf=json.loads((folder/'manifest.json').read_text());assert mf['status']=='frozen_before_inference'
 assert sha(Path(__file__))==mf['runner_sha256'],'Runner changed after benchmark freeze'
 for f,k in [('questions.jsonl','questions_sha256'),('rubric.json','rubric_sha256')]:assert sha(folder/f)==mf[k]
 qs=[json.loads(s)for s in (folder/'questions.jsonl').read_text().splitlines()];assert len(qs)==mf['question_count'];base=ROOT/'models/qwen3-4b-base';out=folder/'runs/original';out.mkdir(parents=True,exist_ok=True);index=json.loads((base/'model.safetensors.index.json').read_text());weights=sorted(set(index['weight_map'].values()))
 settings={'model':'Qwen3-4B-Instruct-2507','adapter_path':None,'base_path':str(base),'model_config_sha256':sha(base/'config.json'),'model_index_sha256':sha(base/'model.safetensors.index.json'),'tokenizer_sha256':sha(base/'tokenizer.json'),'weights':{f:{'bytes':(base/f).stat().st_size,'mtime_ns':(base/f).stat().st_mtime_ns}for f in weights},'questions_sha256':sha(folder/'questions.jsonl'),'rubric_sha256':sha(folder/'rubric.json'),'runner_sha256':sha(Path(__file__)),'system':SYSTEM,'max_tokens':128,'temperature':0,'seed':42,'runtime':'MLX','scoring':'Full generated completion; keys and rubric never enter prompts.'}
 settings['software']={'python':sys.version,'platform':platform.platform(),'mlx':version('mlx'),'mlx_lm':version('mlx-lm')}
 meta=out/'metadata.json';existing=json.loads(meta.read_text())if meta.exists()else None
 if existing:assert existing['settings']==settings
 if existing and existing['status']=='completed':print('Already completed');return
 prior=recover(out/'responses.jsonl');done={r['id']for r in prior};assert len(done)==len(prior)and done<={q['id']for q in qs};started=existing.get('started_at')if existing else datetime.datetime.now(datetime.timezone.utc).isoformat();atomic(meta,{'settings':settings,'status':'running','started_at':started,'completed_items':len(done),'pid':os.getpid()})
 model,tok=load(str(base));model.eval();mx.random.seed(42)
 with (out/'responses.jsonl').open('a')as f:
  for q in qs:
   if q['id']in done:continue
   prompt=tok.apply_chat_template([{'role':'system','content':SYSTEM},{'role':'user','content':user_prompt(q)}],tokenize=False,add_generation_prompt=True);t=time.monotonic();pieces=list(stream_generate(model,tok,prompt=prompt,max_tokens=128,sampler=make_sampler(temp=0)));assert pieces
   row={'id':q['id'],'section':q['section'],'answer':''.join(x.text for x in pieces),'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'generation_tokens':pieces[-1].generation_tokens,'finish_reason':pieces[-1].finish_reason,'seconds':time.monotonic()-t};f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno());done.add(q['id']);atomic(meta,{'settings':settings,'status':'running','started_at':started,'completed_items':len(done),'pid':os.getpid()});mx.clear_cache();print(q['id'],len(done),'/',len(qs),flush=True)
 atomic(meta,{'settings':settings,'status':'completed','started_at':started,'finished_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'completed_items':len(done),'responses_sha256':sha(out/'responses.jsonl'),'peak_memory_gb':mx.get_peak_memory()/1e9})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--release',default='v1');main(p.parse_args().release)
