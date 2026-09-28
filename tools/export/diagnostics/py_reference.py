import json, sys, torch
from laya.agent import Agent
from laya.presets import triage_questions, guard_questions, email_questions
torch.use_deterministic_algorithms(True)
model_dir, out = sys.argv[1], sys.argv[2]
states = json.load(open(__import__('os').path.join(__import__('os').path.dirname(__file__), 'states.json')))
sets = {'triage': triage_questions(), 'guard': guard_questions(), 'email': email_questions()}
agent = Agent(model_dir, compile=False, device='cpu')
res = []
for si, s in enumerate(states):
    for qs, q in sets.items():
        r = agent.system_one(s, q)
        for k, a in r['answers'].items():
            p = {'true': a['noul'], 'false': 1 - a['noul']} if a['type'] == 'noul' else a.get('probabilities', {})
            res.append({'id': f'{si}/{qs}/{k}', 'p': p, 'act': (a.get('action') or {}).get('act_probability')})
json.dump(res, open(out, 'w'))
print('rows', len(res))
