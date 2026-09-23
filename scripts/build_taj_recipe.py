"""Freeze reviewed Taj data, group-aware validation split and bounded training order."""
import json,hashlib,random,re,unicodedata,collections,math
from difflib import SequenceMatcher
from pathlib import Path
from transformers import AutoTokenizer
from mlx_lm.tuner.datasets import ChatDataset
from run_taj_holdout import SYSTEM
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'data/taj_recipe_v1';E=ROOT/'experiments/taj_recipe_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def norm(s):return re.findall(r'\w+',unicodedata.normalize('NFKC',s).lower())
def dump(p,rows):p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
def main():
    assert not (D/'manifest.json').exists(),'Never overwrite a frozen data release'
    approval=json.loads((E/'data_approval.json').read_text());assert approval['status']=='approved_for_training_build'
    source=D/'teacher/teacher_revised.jsonl';assert sha(source)==approval['teacher_revised_sha256']
    rows=read(source);assert len(rows)==len({r['id'] for r in rows})
    benchmark=ROOT/'evaluation/taj_holdout/releases/v1';bm=json.loads((benchmark/'manifest.json').read_text());assert sha(benchmark/'questions.jsonl')==bm['questions_sha256']
    test=read(benchmark/'questions.jsonl');testnorm={q['id']:norm(q['question']) for q in test}
    keep=[];quarantine=[];seen=set()
    for r in rows:
        assert r['claim_ids'] and 'V018' not in r['claim_ids'] and r['source_urls']
        words=norm(r['question']);key=' '.join(words)
        if key in seen:quarantine.append({'id':r['id'],'reason':'exact_duplicate_selected_question'});continue
        overlaps=[]
        for id,other in testnorm.items():
            ratio=SequenceMatcher(None,words,other,autojunk=False).ratio()
            if words==other or ratio>=0.92:overlaps.append({'benchmark_id':id,'token_sequence_similarity':ratio})
        if overlaps:quarantine.append({'id':r['id'],'reason':'exact_or_near_evaluation_wording','matches':overlaps});continue
        seen.add(key);keep.append(r)
    tokenizer=AutoTokenizer.from_pretrained(str(ROOT/'models/qwen3-4b-base'),local_files_only=True)
    def chat(r):return {'messages':[{'role':'system','content':SYSTEM},{'role':'user','content':r['question']},{'role':'assistant','content':r['answer']}]}
    ds=ChatDataset([],tokenizer,mask_prompt=True);encoded={};eligible=[]
    for r in keep:
        tokens,offset=ds.process(chat(r))
        if not(0<offset<len(tokens)<=512):quarantine.append({'id':r['id'],'reason':'invalid_or_overlength_chat','length':len(tokens)});continue
        encoded[r['id']]={'tokens':len(tokens),'answer_tokens':len(tokens)-offset};eligible.append(r)
    # Group teacher families plus lexical near-duplicates, without holding out Taj facts.
    parent=list(range(len(eligible)))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(i,j):parent[find(j)]=find(i)
    words=[norm(r['question']) for r in eligible]
    for i,a in enumerate(eligible):
        for j in range(i):
            b=eligible[j]
            if a['family_id']==b['family_id']:union(i,j)
            elif set(a['claim_ids'])&set(b['claim_ids']) and SequenceMatcher(None,words[i],words[j],autojunk=False).ratio()>=0.86:union(i,j)
    groups=collections.defaultdict(list)
    for i,r in enumerate(eligible):groups[find(i)].append(r)
    grouped=list(groups.values());rng=random.Random(20260916);rng.shuffle(grouped)
    remaining=collections.Counter(c for r in eligible for c in set(r['claim_ids']));total_topic=collections.Counter(r['topic'] for r in eligible)
    valid_topic=collections.Counter();validgroups=set()
    for index,group in enumerate(grouped):
        counts=collections.Counter(c for r in group for c in set(r['claim_ids']))
        topics=collections.Counter(r['topic'] for r in group)
        wanted=any(valid_topic[t]<round(total_topic[t]*0.15) for t in topics)
        if wanted and all(remaining[c]>n for c,n in counts.items()):
            validgroups.add(index);remaining.subtract(counts);valid_topic.update(topics)
    splits={'train':[],'valid':[]};members=[]
    for index,group in enumerate(grouped):
        split='valid' if index in validgroups else 'train';family=f'GROUP-{index:04d}'
        for r in group:
            r={**r,'split':split,'split_family_id':family};splits[split].append(r);members.append(r)
    assert min(len(v) for v in splits.values())>=32
    assert {r['split_family_id'] for r in splits['train']}.isdisjoint({r['split_family_id'] for r in splits['valid']})
    assert {c for r in splits['valid'] for c in r['claim_ids']}<={c for r in splits['train'] for c in r['claim_ids']}
    for split,rs in splits.items():
        rng.shuffle(rs);dump(D/f'{split}.jsonl',[chat(r) for r in rs]);dump(D/f'{split}.audit.jsonl',rs)
    dump(D/'quarantine.jsonl',quarantine)
    schedule=[];ordering=random.Random(42);epoch_ends=[]
    for epoch in range(3):
        indices=list(range(len(splits['train'])));ordering.shuffle(indices)
        schedule.extend(indices[i:i+4] for i in range(0,len(indices),4));epoch_ends.append(len(schedule))
    (D/'schedule.json').write_text(json.dumps(schedule))
    cfg={'model':'models/qwen3-4b-base','fine_tune_type':'lora','adapter_path':'runs/taj_recipe_v1/adapters','data':'data/taj_recipe_v1','batch_size':4,'grad_accumulation_steps':1,'num_layers':16,'lora_parameters':{'rank':16,'scale':32.0,'dropout':0.0},'optimizer':'adam','learning_rate':2e-5,'lr_schedule':None,'epochs':3,'iters':len(schedule),'max_seq_length':512,'mask_prompt':True,'loss':'assistant_suffix_only_excluding_padding','grad_checkpoint':True,'seed':42,'steps_per_eval':100,'save_every':50,'checkpoint_selection':'minimum_validation_loss_among_trained_scheduled_checkpoints'}
    config=ROOT/'configs/taj_recipe_v1.json';config.write_text(json.dumps(cfg,indent=2))
    total=sum(encoded[splits['train'][i]['id']]['answer_tokens'] for batch in schedule for i in batch)
    counts={split:{'rows':len(rs),'topics':dict(collections.Counter(r['topic'] for r in rs)),'claims':len({c for r in rs for c in r['claim_ids']}),'families':len({r['split_family_id'] for r in rs}),'unpadded_chat_tokens':sum(encoded[r['id']]['tokens'] for r in rs),'assistant_tokens':sum(encoded[r['id']]['answer_tokens'] for r in rs)} for split,rs in splits.items()}
    manifest={'status':'frozen_for_training','teacher_revised_sha256':sha(source),'data_approval_sha256':sha(E/'data_approval.json'),'counts':counts,'reviewed_candidate_count':len(rows),'quarantined':len(quarantine),'epochs':3,'steps':len(schedule),'epoch_end_steps':epoch_ends,'scheduled_examples':sum(map(len,schedule)),'scheduled_assistant_tokens':total,'training_token_definition':'Indices from first assistant-suffix token to last real token; no padding targets. Includes chat closing tokens.','hashes':{f:sha(D/f) for f in ['train.jsonl','valid.jsonl','train.audit.jsonl','valid.audit.jsonl','schedule.json','quarantine.jsonl']},'config_sha256':sha(config),'runner_sha256':sha(ROOT/'scripts/train_taj_recipe.py'),'recovery_helper_sha256':sha(ROOT/'scripts/checkpoint_state.py'),'benchmark_questions_sha256':bm['questions_sha256'],'split_seed':20260916,'train_order_seed':42,'split_method':'15percent per-topic target by teacher/lexical-near-duplicate families; retain representation of every selected claim in training. Validation measures new wording on taught facts.','limitations':['Not unseen-fact evaluation','Lexical grouping does not guarantee all semantic paraphrases are grouped','Existing96item benchmark already inspected during development','Underlying facts intentionally overlap','No unrelated-history retention objective']}
    (D/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
