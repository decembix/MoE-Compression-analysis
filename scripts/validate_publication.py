"""Validate local links, preserved CSV fields and viewer buffers."""
from pathlib import Path
import re,json,csv,hashlib,base64
R=Path(__file__).resolve().parents[1]
def main():
 files=[p for p in R.rglob('*') if p.is_file() and '.git' not in p.parts and 'reproduced' not in p.parts]
 for p in files:
  if p.suffix in ['.pt','.safetensors']:raise AssertionError(p)
  if p.suffix in ['.md','.html']:
   s=p.read_text()
   links=re.findall(r'\]\(([^)]+)\)',s) if p.suffix=='.md' else re.findall(r'(?:href|src)="([^"\n]+)"',s)
   for target in links:
    if target.startswith(('http:','https:','data:','#','mailto:')):continue
    target=target.split('#')[0].split('?')[0];q=p.parent/target
    assert q.exists(),(str(p.relative_to(R)),target)
   assert '/data/xyzysh' not in s and '/home/xyzysh' not in s,p
 manifest=json.loads((R/'results/manifest.json').read_text())
 for row in manifest['artifacts']:assert hashlib.sha256((R/row['published']).read_bytes()).hexdigest()==row['published_sha256'],row['published']
 viewer=(R/'docs/viewer/index.html').read_text();payload=re.search(r'<script id="payload" type="application/json">(.*?)</script>',viewer,re.S).group(1);data=json.loads(payload);assert len(data['layers'])==16
 for l in data['layers']:
  for a in l['arrays'].values():assert len(base64.b64decode(a['base64']))==a['shape'][0]*a['shape'][1]*4
 print(f'PASS: {len(files)} files; Markdown/HTML local links; original CSV hashes;16-layer viewer buffers; no model weights/caches.')
if __name__=='__main__':main()
