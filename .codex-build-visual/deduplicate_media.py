from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from lxml import etree
import hashlib, posixpath, sys
src,dst=map(Path,sys.argv[1:3])
with ZipFile(src) as zin:
 data={i.filename:zin.read(i.filename) for i in zin.infolist()}
 first={}; replacements={}
 for name,blob in data.items():
  if not name.startswith('ppt/media/'):continue
  digest=hashlib.sha256(blob).hexdigest()
  if digest in first:replacements[name]=first[digest]
  else:first[digest]=name
 for name,blob in list(data.items()):
  if not name.endswith('.rels'):continue
  root=etree.fromstring(blob)
  base=posixpath.dirname(name).replace('/_rels','')
  changed=False
  for rel in root:
   if rel.get('TargetMode')=='External':continue
   target=posixpath.normpath(posixpath.join(base,rel.get('Target',''))).lstrip('/')
   if target in replacements:
    rel.set('Target',posixpath.relpath(replacements[target],base));changed=True
  if changed:data[name]=etree.tostring(root,xml_declaration=True,encoding='UTF-8',standalone=True)
 root=etree.fromstring(data['[Content_Types].xml'])
 for el in list(root):
  if el.get('PartName','').lstrip('/') in replacements:root.remove(el)
 data['[Content_Types].xml']=etree.tostring(root,xml_declaration=True,encoding='UTF-8',standalone=True)
 with ZipFile(dst,'w',ZIP_DEFLATED) as zout:
  for name,blob in data.items():
   if name not in replacements:zout.writestr(name,blob)
 print(f'Deduplicated {len(replacements)} repeated media objects; {src.stat().st_size} -> {dst.stat().st_size} bytes')
