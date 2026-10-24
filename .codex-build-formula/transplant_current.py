from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
from lxml import etree as E
import posixpath,sys
current,generated,out=map(Path,sys.argv[1:])
rels_ns='http://schemas.openxmlformats.org/package/2006/relationships'
a_ns='http://schemas.openxmlformats.org/drawingml/2006/main'
p_ns='http://schemas.openxmlformats.org/presentationml/2006/main'
with ZipFile(current) as src,ZipFile(generated) as gen:
 slide=E.fromstring(src.read('ppt/slides/slide14.xml'))
 removed=0
 for sp in list(slide.iter('{'+p_ns+'}sp')):
  content=''.join(sp.xpath('.//a:t/text()',namespaces={'a':a_ns})).strip()
  if content=='14':
   sp.getparent().remove(sp);removed+=1
 if removed!=1:raise RuntimeError(f'Expected one old page number on customized slide, found {removed}')
 rel=E.fromstring(src.read('ppt/slides/_rels/slide14.xml.rels'))
 genrel=E.fromstring(gen.read('ppt/slides/_rels/slide16.xml.rels'))
 dest={r.get('Type').split('/')[-1]:r.get('Target') for r in genrel}
 media={}
 for r in rel:
  kind=r.get('Type').split('/')[-1]
  if kind=='image':
   old=posixpath.normpath(posixpath.join('ppt/slides',r.get('Target')))
   new='ppt/media/user_slide14_'+Path(old).name
   media[new]=src.read(old)
   r.set('Target','../media/'+Path(new).name)
  elif kind in ('notesSlide','slideLayout'):
   r.set('Target',dest[kind])
  else:raise RuntimeError(f'Unrecognized relationship {kind}')
 note_xml=src.read('ppt/notesSlides/notesSlide14.xml')
 note_rel=E.fromstring(src.read('ppt/notesSlides/_rels/notesSlide14.xml.rels'))
 for r in note_rel:
  if r.get('Type').endswith('/slide'):r.set('Target','../slides/slide16.xml')
 replaced={
  'ppt/slides/slide16.xml':E.tostring(slide,xml_declaration=True,encoding='UTF-8',standalone=True),
  'ppt/slides/_rels/slide16.xml.rels':E.tostring(rel,xml_declaration=True,encoding='UTF-8',standalone=True),
  'ppt/notesSlides/notesSlide16.xml':note_xml,
  'ppt/notesSlides/_rels/notesSlide16.xml.rels':E.tostring(note_rel,xml_declaration=True,encoding='UTF-8',standalone=True),
 }
 with ZipFile(out,'w',ZIP_DEFLATED) as z:
  for item in gen.infolist():z.writestr(item,replaced.get(item.filename,gen.read(item.filename)))
  for name,data in media.items():z.writestr(name,data)
print('Kept customized slide and notes; removed its old page number; imported',len(media),'images')
