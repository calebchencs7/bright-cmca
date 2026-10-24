from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from lxml import etree as E
from copy import deepcopy
import re
import sys
src=Path(sys.argv[1])
out=Path(sys.argv[2])
ns='http://schemas.openxmlformats.org/drawingml/2006/main'
q=lambda name: '{'+ns+'}'+name
count=0
with ZipFile(src) as zin, ZipFile(out,'w',ZIP_DEFLATED) as zout:
 for item in zin.infolist():
  data=zin.read(item.filename)
  if item.filename.startswith('ppt/') and item.filename.endswith('.xml'):
   root=E.fromstring(data)
   for n in root.iter():
    if n.tag in [q('latin'),q('ea'),q('cs')]:
     n.set('typeface','Microsoft YaHei' if n.tag==q('ea') else 'Times New Roman'); count+=1
    elif n.tag==q('font') and n.get('script') in ('Hans','Hant','Jpan','Hang'):
     n.set('typeface','Microsoft YaHei')
   for n in root.iter():
    if n.tag in [q('rPr'),q('defRPr'),q('endParaRPr')]:
     for tag,font in [('latin','Times New Roman'),('ea','Microsoft YaHei'),('cs','Times New Roman')]:
      child=n.find(q(tag))
      if child is None:
       child=E.SubElement(n,q(tag))
      child.set('typeface',font)
   if item.filename.startswith('ppt/slides/'):
    pattern=re.compile(r'(?<![A-Za-z])([A-Za-z])_(min|dec|dam|[blsijB])(?![A-Za-z])')
    for run in list(root.iter(q('r'))):
     t=run.find(q('t'))
     if t is None or not t.text: continue
     matches=list(pattern.finditer(t.text))
     if not matches: continue
     parts=[]; last=0
     for m in matches:
      if m.start()>last: parts.append((t.text[last:m.start()],False))
      parts.extend([(m.group(1),False),(m.group(2),True)])
      last=m.end()
     if last<len(t.text):parts.append((t.text[last:],False))
     parent=run.getparent(); idx=parent.index(run); prop=run.find(q('rPr'))
     for k,(value,sub) in enumerate(parts):
      nr=E.Element(q('r'));np=deepcopy(prop) if prop is not None else E.Element(q('rPr'))
      if sub:
       np.set('baseline','-25000')
       if np.get('sz'):np.set('sz',str(round(int(np.get('sz'))*.75)))
      nr.append(np);nt=E.SubElement(nr,q('t'));nt.text=value
      nt.set('{http://www.w3.org/XML/1998/namespace}space','preserve')
      parent.insert(idx+k,nr)
     parent.remove(run)
   data=E.tostring(root,xml_declaration=True,encoding='UTF-8',standalone=True)
  zout.writestr(item,data)
print('Updated font declarations:',count)
