import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { execFileSync } from 'node:child_process';
import { Presentation, PresentationFile } from '@oai/artifact-tool';

const workspaceDir = '/Users/haochen/Documents/Development/bright-cmca';
const skillDir = '/Users/haochen/.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11127/skills/presentations';
const buildDir = path.join(workspaceDir, '.codex-build-formula');
const outputDir = path.join(workspaceDir, 'output', 'pptx');
const finalPath = path.join(outputDir, '组会汇报_202610_陈昊_CMCA_DACutMix_方法公式图解_无页码版.pptx');
const imageDir = path.join(workspaceDir, 'latex', 'images');
const imageBytes = new Map();
for (const name of ['Technology Roadmap.png','CMCA.png','DA-CutMix.png','dacutmix_example.png','qualitative_results.png']) {
  const bytes = await fs.readFile(path.join(imageDir,name));
  imageBytes.set(name, bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
}
const py = '/Users/haochen/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3';
const nativeTableSlides=[];
const extraBytes=new Map();
for(const name of await fs.readdir(path.join(buildDir,'assets'))){
 if(!/\.(png|jpg)$/.test(name)) continue;
 const b=await fs.readFile(path.join(buildDir,'assets',name));
 extraBytes.set(name,b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength));
}
const W = 1280, H = 720;
const C = {
  dark:'#4B5563', lime:'#DEFF9B', green:'#12B981', navy:'#1E2761',
  red:'#C8523E', gold:'#E0A84B', purple:'#7953A9', pale:'#F7F7FA',
  ink:'#36454F', muted:'#647480', line:'#DDDDE3', white:'#FFFFFF',
  lightGreen:'#EAF6F3', lightBlue:'#EEF2FB', lightRed:'#FBF0EE'
};
const FONT = 'Microsoft YaHei';
const ppt = Presentation.create({slideSize:{width:W,height:H}});

function rect(slide,x,y,w,h,fill,line='none',lw=0){
  return slide.shapes.add({geometry:'rect',position:{left:x,top:y,width:w,height:h},fill,
    line:{style:'solid',fill:line,width:lw}});
}
function txt(slide,s,x,y,w,h,size=24,color=C.ink,bold=false,opts={}){
  const sh=slide.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',
    line:{style:'solid',fill:'none',width:0}});
  sh.text=String(s);
  sh.text.style={fontSize:size,color,bold,typeface:FONT,alignment:opts.align||'left',
    verticalAlignment:opts.valign||'top',autoFit:'none',wrap:'square',
    insets:opts.insets??0,lineSpacing:opts.lineSpacing??1.06};
  return sh;
}
function addHeader(slide,title,subtitle='',number=''){
  slide.background.fill=C.white;
  rect(slide,0,0,W,116,C.pale);
  rect(slide,62,30,7,54,C.green);
  txt(slide,title,90,29,1110,47,31,C.ink,true);
  if(subtitle) txt(slide,subtitle,91,80,1110,24,14,C.muted,false);
  rect(slide,62,116,1156,2,C.line);
  txt(slide,'CMCA + DA-CutMix  ·  组会汇报  ·  2026.10',62,682,800,17,12,C.muted);
}
function addSection(label,title,en,num){
  const s=ppt.slides.add();s.background.fill=C.dark;
  txt(s,`PART ${label}`,64,230,500,28,17,'#A7B2BD',true);
  txt(s,title,64,284,1080,80,46,C.white,true);
  rect(s,0,407,W,2,C.lime);
  txt(s,en,64,437,1080,40,23,C.white);
  rect(s,64,510,95,5,C.lime);
  return s;
}
function note(slide,text){slide.speakerNotes.text=text;}
function img(slide,file,x,y,w,h,alt,fit='contain',crop){
  return slide.images.add({blob:imageBytes.get(file),contentType:'image/png',alt,fit,
    position:{left:x,top:y,width:w,height:h},...(crop?{crop}:{})});
}
function line(slide,x,y,w,color=C.line,thick=2){rect(slide,x,y,w,thick,color);}
function label(slide,n,title,body,x,y,w,color=C.green){
  rect(slide,x,y,5,108,color);
  txt(slide,n,x+17,y+2,42,25,16,color,true);
  txt(slide,title,x+62,y+1,w-62,30,22,C.ink,true);
  txt(slide,body,x+62,y+38,w-68,65,17,C.ink);
}
function asset(slide,file,x,y,w,h,alt,fit='contain',crop){
 return slide.images.add({blob:extraBytes.get(file),contentType:file.endsWith('.jpg')?'image/jpeg':'image/png',alt,fit,
 position:{left:x,top:y,width:w,height:h},...(crop?{crop}:{})});
}
function eq(slide,key,x,y,w,h,tag=''){
 asset(slide,key+'.png',x,y,w-(tag?80:0),h,'论文公式 '+key);
 if(tag) txt(slide,tag,x+w-77,y+h/2-15,77,34,21,C.muted,false,{align:'right'});
}
function defs(slide,rows,y,opt={}){
 const x=opt.x??80,w=opt.w??1120,keyW=opt.keyW??260,rowH=opt.rowH??66,size=opt.size??22;
 rows.forEach(([key,meaning],i)=>{
   const yy=y+i*rowH;
   asset(slide,'sym_'+key+'.png',x,yy+5,keyW-20,Math.min(43,rowH-10),'符号 '+key);
   txt(slide,meaning,x+keyW,yy+7,w-keyW,rowH-9,size,C.ink);
 });
}
function table(slide,values,x,y,w,h,widths=null,fs=17,highlightRows=[]){
  nativeTableSlides.push(ppt.slides.items.length);
  const t=slide.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width:w,height:h,
    values,...(widths?{columnWidths:widths}:{})});
  t.styleOptions={headerRow:true,bandedRows:false};
  t.borders.assign({style:'solid',fill:C.line,width:0.6});
  for(let r=0;r<values.length;r++) for(let c=0;c<values[0].length;c++){
    const cell=t.getCell(r,c);cell.fill=r===0?C.navy:(highlightRows.includes(r)?C.lightGreen:C.white);
    cell.text.style={fontSize:fs,typeface:FONT,color:r===0?C.white:C.ink,bold:r===0||c===0,
      alignment:c===0?'left':'center',verticalAlignment:'middle',insets:6,autoFit:'none'};
  }
  return t;
}
function featureGrid(slide,x,y,size,n,seed=0,highlight=false){
 const palettes=[
  ['#EEF2FA','#C8D8F0','#89A9D6','#4E73AF'],
  ['#EFF8F5','#BDE9D8','#72C5A5','#279A72'],
  ['#F5F0F9','#DEC9ED','#B18ACD','#805AA7']
 ];
 const pal=palettes[seed%3], cell=size/n;
 for(let row=0;row<n;row++) for(let col=0;col<n;col++){
  const level=(row*3+col*5+seed*7+(row*col)%5)%4;
  rect(slide,x+col*cell,y+row*cell,cell-0.5,cell-0.5,pal[level]);
 }
 rect(slide,x,y,size,size,'none',C.navy,1.3);
 if(highlight) rect(slide,x+4*cell,y+3*cell,cell,cell,'none',C.gold,3);
}

// 1 — Cover
{
 const s=ppt.slides.add();s.background.fill=C.dark;
 txt(s,'RESEARCH PROGRESS · 2026',64,74,720,28,17,'#C7D0D9',true);
 txt(s,'基于跨模态变化注意力与\n损伤感知数据增强的建筑损伤评估',64,151,1100,155,45,C.white,true);
 rect(s,0,344,W,2,C.lime);
 txt(s,'Multimodal Building Damage Assessment With Cross-Modal Attention and Damage-Aware Augmentation',64,376,1130,64,22,C.white);
 txt(s,'汇报人：陈昊     导师：温家洪、廉洁',64,558,900,30,20,C.white);
 txt(s,'2026年10月',64,603,400,25,17,C.lime);
 note(s,'基于 2026 年 10 月论文稿件。汇报重点：方法、六骨干消融、逐事件差异与失败案例。');
}
// 2 — Outline
{
 const s=ppt.slides.add();addHeader(s,'汇报提纲','Research progress and current manuscript',2);
 const items=[['01','研究背景与问题'],['02','方法：CMCA 与 DA-CutMix'],['03','实验结果与分析'],['04','局限与下一步工作']];
 items.forEach((it,i)=>{const y=184+i*111;txt(s,it[0],82,y,75,40,30,C.green,true);txt(s,it[1],170,y,900,42,30,C.ink,true);line(s,170,y+61,930,C.line,2);});
}
// 3 — divider
addSection('01','研究背景与问题','Background & research questions',3);
// 4 — Background
{
 const s=ppt.slides.add();addHeader(s,'任务背景：灾后光学影像可能缺失','Pre-event optical + post-event SAR for building damage assessment',4);
 txt(s,'灾前光学',87,173,430,43,34,C.navy,true);
 txt(s,'建筑几何、纹理与空间布局',87,225,430,47,22,C.ink);
 txt(s,'灾后 SAR',685,173,430,43,34,C.red,true);
 txt(s,'不依赖日照，受云烟影响较小',685,225,480,47,22,C.ink);
 rect(s,86,304,1090,3,C.line);
 txt(s,'核心任务',86,345,200,36,20,C.green,true);
 txt(s,'利用时相与模态均不同的一对影像，预测每个像素的建筑损伤等级。',86,385,1090,80,29,C.ink,true);
 txt(s,'类别：背景 / 完好 / 受损 / 毁坏',86,535,1090,38,23,C.muted);
 note(s,'论文引言指出，灾后云、烟和光照不足会影响光学观测；SAR 为灾后观测提供替代。这里讨论的是灾前 RGB 光学与灾后单波段 SAR 振幅，SAR 单波段在模型接口中重复为三通道。');
}
// Restored dataset background from the reference presentation.
{
 const s=ppt.slides.add();addHeader(s,'数据集发展：从 ABCD、xBD 到 BRIGHT','From optical imagery to multimodal damage benchmarks');
 table(s,[
  ['年份 / 数据集','影像与任务','规模与覆盖','公开性'],
  ['2017  ABCD','图像级损伤识别','以海啸场景为主','—'],
  ['2019  xBD','灾前 / 灾后光学','约 70 万建筑；15 事件 / 6 类灾害','公开'],
  ['2021  BDD','光学 + SAR','约 12 万建筑；9 事件 / 3 类灾害','未公开'],
  ['2025  BRIGHT','灾前光学 + 灾后 SAR','约 38 万建筑；14 事件 / 7 类灾害','公开']
 ],72,177,1136,340,[235,300,420,181],19,[4]);
 txt(s,'BRIGHT 将多模态影像、像素级损伤标注与多灾害事件整合到统一基准。',80,556,1110,66,25,C.navy,true);
 note(s,'保留原 PPT 第 9 页的数据集演进介绍。规模为原稿的近似口径。重点说明研究任务如何由光学双时相扩展至灾前光学/灾后 SAR。');
}
{
 const s=ppt.slides.add();addHeader(s,'BRIGHT：全球事件分布与数据规模','Global distribution and very-high-resolution observations');
 asset(s,'bright_overview.jpg',72,167,805,443,'原 PPT 中 BRIGHT 全球事件分布图','cover',{left:0,top:0,right:0.56,bottom:0.59});
 const stats=[['14','灾害事件'],['23','研究区域'],['384,596','建筑实例标注'],['0.3–1 m','空间分辨率']];
 stats.forEach((v,i)=>{txt(s,v[0],923,169+i*113,294,48,34,i?C.navy:C.green,true);txt(s,v[1],926,220+i*113,280,28,19,C.muted);});
 note(s,'图保留自原 PPT 第 10 页，原始来源为 BRIGHT 数据集论文 Chen et al., ESSD 2025。384,596 指建筑实例，其损伤具有像素级标注。');
}
{
 const s=ppt.slides.add();addHeader(s,'BRIGHT：不同灾害的光学、SAR 与损伤标注','Examples from earthquakes, wildfire, flood, explosions and a hurricane');
 asset(s,'bright_overview.jpg',258,145,758,514,'BRIGHT 原图中的前十个事件影像与标签样例','cover',{left:0.443,top:0.012,right:0,bottom:0.315});
 txt(s,'每组样例',74,218,162,32,21,C.green,true);
 txt(s,'灾前光学\n\n灾后 SAR\n\n损伤标签',74,275,170,198,21,C.ink);
 txt(s,'相同灾种也具有\n不同的建筑布局\n与 SAR 外观',1042,291,184,150,19,C.muted);
 note(s,'图保留自原 PPT 第 10 页的事件缩略图区域。三行依次为灾前光学、灾后 SAR 与参考损伤图。保留原图事件名称、类别色标及全部前十个场景。');
}
{
 const s=ppt.slides.add();addHeader(s,'BRIGHT：地震、火山、飓风与冲突样例','Earthquake, volcanic eruption, hurricane and conflict imagery');
 asset(s,'bright_overview.jpg',83,173,1114,366,'摩洛哥地震、拉帕尔马火山、墨西哥飓风、乌克兰冲突样例','cover',{left:0.443,top:0.683,right:0,bottom:0});
 txt(s,'同一损伤等级的外观，会随灾害类型、建筑结构与观测条件变化。',89,584,1090,67,25,C.ink,true);
 note(s,'图保留自原 PPT 第 10 页的最后四个事件：摩洛哥地震、拉帕尔马火山喷发、阿卡普尔科飓风、马里乌波尔冲突。该页补齐原图的 14 事件样例。');
}
{
 const s=ppt.slides.add();addHeader(s,'BRIGHT：模态差异与类别不平衡','Pixel statistics, building sizes and class distribution');
 asset(s,'bright_overview.jpg',70,153,690,509,'原 PPT 的 BRIGHT 模态、建筑尺度与类别统计图','cover',{left:0,top:0.43,right:0.56,bottom:0});
 txt(s,'建筑像素只占 14.1%',803,179,410,42,27,C.navy,true);
 txt(s,'背景占 85.9%，总体像素精度容易受到背景主导。',803,231,398,70,21,C.ink);
 txt(s,'建筑像素内部',803,347,410,35,22,C.green,true);
 txt(s,'完好 82.8%\n受损 10.7%\n毁坏 6.5%',803,394,398,123,24,C.ink,true);
 txt(s,'DA-CutMix 重点增加受损与毁坏类别的训练曝光。',803,561,398,66,21,C.red,true);
 note(s,'保留原 PPT 第 10 页的统计图。85.9%/14.1% 的分母是全部像素；82.8%/10.7%/6.5% 的分母是建筑像素，两组比例不可混用。');
}

// 5 — Dataset and challenge
{
 const s=ppt.slides.add();addHeader(s,'BRIGHT 数据集与主要难点','14 events · 23 regions · four semantic classes',5);
 const stats=[['14','灾害事件'],['23','区域'],['2,917','训练样本'],['449 / 877','验证 / 测试']];
 stats.forEach((v,i)=>{const x=80+i*293;txt(s,v[0],x,166,260,55,40,i===0?C.green:C.navy,true);txt(s,v[1],x,225,260,30,18,C.muted);});
 line(s,80,294,1120);
 label(s,'Q1','跨模态对应难','光学反射与 SAR 后向散射的物理机制不同，配准不等于特征可直接比较。',82,332,1080,C.red);
 label(s,'Q2','损伤类别样本少','受损与毁坏像素少于背景和完好建筑，且不同灾害事件的外观差异明显。',82,478,1080,C.green);
 note(s,'BRIGHT 标准监督划分：训练 2,917，验证 449，测试 877。包含 14 个事件、23 个区域和四类像素标注。数据来源：论文第 IV-A 节。');
}
// 6 divider
addSection('02','方法：CMCA 与 DA-CutMix','Feature interaction and damage-aware augmentation',6);
// Technical method section with local notation definitions.
{
 const s=ppt.slides.add();addHeader(s,'问题定义：光学–SAR 四类建筑损伤分割','输入、预测目标与符号约定');
 eq(s,'eq1',83,150,1110,85,'（1）');
 eq(s,'inputs',90,250,1095,60);
 defs(s,[['xy','灾前 RGB 光学与灾后 SAR 振幅；上标 o、s 分别表示两种模态。'],['hw','输入影像的高、宽；SAR 单波段复制为三通道以匹配编码器接口。'],['yy','Y 是监督标签，Ŷ 是预测图；0/1/2/3 对应背景、完好、受损、毁坏。'],['ftheta','f 是分割模型，θ 为全部可学习参数；训练时用 Y 监督预测。']],338,{rowH:66});
 txt(s,'后续特征图尺寸用小写 h、w；公式统一省略批次维。',85,626,1110,35,22,C.navy,true);
 note(s,'本文式（1）、III-A。强调空间配准只保证数据在共同网格上，光学反射与 SAR 后向散射仍有不同统计特性。X 为三通道影像，Y 与 Ŷ 是 H×W 的离散标签图。实数域 R 表示输入张量是实数；集合 {0,1,2,3} 表示每个输出像素的类别取值。帽子表示预测，不表示增强。SAR 复制三通道不等于三个极化观测。');
}
{
 const s=ppt.slides.add();addHeader(s,'总体框架：双分支表征、定向交互与损伤监督','CMCA 改变特征融合；DA-CutMix 改变训练样本分布');
 img(s,'Technology Roadmap.png',72,145,1136,359,'论文总体框架图');
 txt(s,'模型路径',82,528,170,35,24,C.navy,true);
 txt(s,'独立编码获得成对特征，CMCA 融合后经适配器连接原解码器。',265,528,930,65,23);
 txt(s,'训练路径',82,606,170,35,24,C.red,true);
 txt(s,'几何增强后进行跨事件区域混合，以同步替换的标签提供损伤监督。',265,606,930,65,23);
 note(s,'本文 Figure 1、III-B。网络的输入是配准后的灾前光学与灾后 SAR。DA-CutMix 位于几何增强之后、归一化之前，只在训练时使用。双分支编码器提取不同传感器的表示；选定语义层进入 CMCA；适配器匹配下游接口。解码器或预测头输出四类 logits，选取得分最高的类别。具有 skip 的模型另用浅层融合信息恢复细节。');
}
// Concepts before the encoder equations.
{
 const s=ppt.slides.add();addHeader(s,'特征图：影像如何变成多通道数值网格','以论文中的 UNet 光学分支为例；彩色网格只表示特征值，不是真实激活图');
 txt(s,'输入影像',83,159,240,30,22,C.navy,true);
 txt(s,'640 × 640 × 3',83,188,250,26,18,C.muted);
 asset(s,'optical_flood.png',83,224,230,230,'论文洪水样例的灾前光学影像');
 rect(s,83,224,230,230,'none',C.line,1.2);
 txt(s,'→',322,303,58,57,48,C.green,true);
 txt(s,'Encoder',372,291,156,34,24,C.navy,true);
 txt(s,'提取并下采样',372,332,155,28,17,C.muted);
 txt(s,'→',510,303,50,57,48,C.green,true);
 txt(s,'同一空间位置，不同通道的特征值',568,161,595,32,21,C.navy,true);
 featureGrid(s,566,225,111,8,0,true);
 featureGrid(s,704,225,111,8,1,true);
 txt(s,'···',825,259,56,46,35,C.muted,true);
 featureGrid(s,883,225,111,8,2,true);
 txt(s,'通道 1',580,353,111,26,18,C.ink);
 txt(s,'通道 2',715,353,111,26,18,C.ink);
 txt(s,'通道 512',884,353,135,26,18,C.ink);
 txt(s,'金色方格表示各通道中的同一网格位置',570,406,582,32,19,C.ink);
 line(s,83,485,1115);
 txt(s,'空间网格：40 × 40',83,510,475,36,23,C.navy,true);
 txt(s,'每个格点：512 个通道值',593,510,595,36,23,C.navy,true);
 txt(s,'论文中的 UNet：640 ÷ 16 = 40；光学和 SAR 两路各有一张 512 × 40 × 40 的深层特征图。',83,577,1110,65,20,C.ink);
 note(s,'视觉示意。左图裁自论文定性结果图中的洪水样例灾前光学影像。右侧 8×8 小网格仅示意特征图的通道与空间索引，不是模型真实激活值，也不是实际的 40×40 全部格点。每一通道是一张 40×40 数值网格；同一空间位置跨 512 个通道的数值组成 512 维特征向量。论文 Table IV 中 UNet 的 CMCA 接入特征通道数为 512，空间尺寸 40×40，训练裁块为 640×640。光学和 SAR 两路分别编码，各输出相应特征。');
}
{
 const s=ppt.slides.add();addHeader(s,'从深层特征到逐像素分类图','以 UNet 式解码路径和论文中的洪水样例说明');
 txt(s,'CMCA 融合特征',78,164,246,32,22,C.navy,true);
 txt(s,'512 × 40 × 40',78,197,246,28,18,C.muted);
 featureGrid(s,88,241,142,8,1,false);
 txt(s,'→',248,285,59,62,48,C.green,true);
 txt(s,'Decoder',331,164,420,32,24,C.navy,true);
 txt(s,'逐步上采样，并结合浅层 skip 特征恢复边界',331,197,610,32,18,C.ink);
 featureGrid(s,331,282,77,7,0,false);
 txt(s,'→',414,286,48,55,38,C.green,true);
 featureGrid(s,470,263,115,7,0,false);
 txt(s,'→',590,286,49,55,38,C.green,true);
 featureGrid(s,648,240,160,8,0,false);
 txt(s,'80',346,382,70,22,17,C.muted);
 txt(s,'160',498,382,80,22,17,C.muted);
 txt(s,'640',704,408,80,22,17,C.muted);
 txt(s,'40 → 80 → 160 → 320 → 640',329,455,491,34,21,C.navy,true);
 txt(s,'→',814,286,52,55,40,C.green,true);
 txt(s,'预测头',849,222,110,30,21,C.navy,true);
 txt(s,'每像素 4 个\n类别得分',846,262,115,70,18,C.ink);
 txt(s,'s₀ s₁ s₂ s₃',845,344,143,31,18,C.ink);
 txt(s,'→',956,286,48,55,40,C.green,true);
 asset(s,'unet_flood.png',1008,225,185,185,'论文洪水样例中 UNet + CMCA + DA-CutMix 的分类预测图');
 rect(s,1008,225,185,185,'none',C.line,1.2);
 txt(s,'像素分类图',1007,164,185,32,22,C.navy,true);
 txt(s,'640 × 640',1020,423,177,28,18,C.muted);
 line(s,83,517,1115);
 txt(s,'同一像素的四个得分取最大值，得到背景、完好、受损或毁坏标签。',83,548,1099,40,21,C.ink,true);
 const legend=[['背景','#FFFFFF',83],['完好','#41BD82',330],['受损','#EAC08D',560],['毁坏','#BD4748',790]];
 legend.forEach(([name,fill,x])=>{rect(s,x,609,25,25,fill,C.line,1);txt(s,name,x+35,607,150,30,18,C.ink);});
 txt(s,'右图为论文中的实际预测样例',994,608,220,30,15,C.muted);
 note(s,'左侧深层融合特征为示意，并非实际激活。CMCA 将光学和 SAR 成对深层特征融合，再通过适配器接回 UNet 原解码器。小网格示意解码器逐级恢复空间尺度，通常结合浅层 skip 特征补充边界。每个像素经过预测头得到四个类别得分，取最高分生成 H×W 的类别图。右图裁自论文定性结果图中的洪水样例 UNet + CMCA + DA-CutMix 预测，与上一页的洪水光学样例对应。彩图颜色按论文图例表示背景、完好、受损、毁坏；示意网格颜色没有类别意义。');
}
{
 const s=ppt.slides.add();addHeader(s,'双分支编码：逐层建立模态特定表征','同一层的两种模态具有可融合的空间尺寸与通道数');
 eq(s,'eq2',85,156,1095,86,'（2）');
 defs(s,[['ml','m 是模态索引；l 是编码阶段，L 为总阶段数。'],['em','E 是该阶段的编码映射；输入为上一阶段的特征。'],['fm0','第 0 层特征就是对应模态的输入影像。'],['fl','F 是第 l 阶段输出的三维特征张量。'],['cl','C 是通道数，h、w 是特征图高宽；下采样后通常小于原图 H、W。']],280,{rowH:61,keyW:292});
 txt(s,'深层特征用于跨模态语义交互；浅层特征可为解码器补充边界信息。',85,617,1110,48,23,C.navy,true);
 note(s,'本文式（2）、III-C。各符号的下标 l 表示第 l 个阶段，不是样本索引。F_l^m 的每个空间位置包含 C_l 维表示。上下两支分别学习光学和 SAR 表征，参数不要求共享。其目的在于先处理传感器差异，再在语义层进行交互；配准后的图像也不应直接按亮度做差。');
}
{
 const s=ppt.slides.add();addHeader(s,'浅层融合：拼接后投影，保留解码所需细节','适用于具有显式跳跃连接的编码器–解码器');
 eq(s,'eq3',88,160,1090,105,'（3）');
 defs(s,[['concat','沿通道维拼接；两路 C_l 通道合为 2C_l，空间高宽保持不变。'],['eta','可学习的 1×1 卷积投影，把拼接通道映射到原解码器所需通道数。'],['skip','第 l 层的融合跳跃特征，送往对应解码阶段。']],313,{rowH:86,keyW:220});
 txt(s,'设计理由：将较浅层的空间细节直接交给解码器，补充深层语义表示。',85,605,1110,55,24,C.navy,true);
 note(s,'本文式（3）。S_l 的通道数由原解码器接口决定，因此不应一律写成 C_l；其空间分辨率仍为 h_l×w_l。η_l 重新组合通道，同时匹配下游维度。F_l^o、F_l^s、l 的含义承接上一页。无显式跳跃连接的骨干不需要这一式，只需暴露成对深层特征。');
}
{
 const s=ppt.slides.add();addHeader(s,'图解式（3）：两路浅层特征如何送入 Decoder','通道拼接改变通道数，1 × 1 投影匹配原解码器接口');
 txt(s,'光学浅层特征',78,169,203,30,22,C.navy,true);
 txt(s,'Cₗ × hₗ × wₗ',78,201,203,27,18,C.muted);
 featureGrid(s,80,258,134,7,0,false);
 txt(s,'+',237,292,53,66,48,C.green,true);
 txt(s,'SAR 浅层特征',294,169,207,30,22,C.navy,true);
 txt(s,'Cₗ × hₗ × wₗ',294,201,207,27,18,C.muted);
 featureGrid(s,296,258,134,7,1,false);
 txt(s,'→',444,302,63,52,43,C.green,true);
 txt(s,'沿通道拼接',510,169,215,30,22,C.navy,true);
 featureGrid(s,550,267,118,7,0,false);
 featureGrid(s,567,284,118,7,1,false);
 txt(s,'2Cₗ × hₗ × wₗ',525,432,210,32,19,C.ink);
 txt(s,'→',712,302,56,52,43,C.green,true);
 txt(s,'1 × 1 投影 ηₗ',764,217,198,37,22,C.navy,true);
 rect(s,798,277,121,100,C.lightBlue,C.navy,1.5);
 txt(s,'通道重组',812,307,103,35,19,C.ink,true);
 txt(s,'→',934,302,53,52,43,C.green,true);
 txt(s,'融合 skip 特征 Sₗ',989,169,238,32,22,C.navy,true);
 featureGrid(s,1018,258,134,7,2,false);
 txt(s,'送入对应的解码阶段',984,415,242,32,19,C.ink);
 line(s,80,501,1115);
 txt(s,'空间网格 hₗ × wₗ 始终对齐；ηₗ 只把 2Cₗ 通道映射到原 Decoder 需要的通道数。',83,537,1100,72,21,C.ink,true);
 note(s,'图解对应论文式（3），所有彩色网格为示意而非真实激活。F_l^o 与 F_l^s 的空间高宽相同，沿通道拼接成为 2C_l×h_l×w_l。η_l 是可学习的 1×1 投影，生成原解码器接口所需通道数的融合跳跃特征 S_l。此操作只适用于带显式 skip 连接的编码器解码器骨干，不是所有六个骨干都具有相同的 skip 路径。');
}
{
 const s=ppt.slides.add();addHeader(s,'CMCA：光学查询驱动的 SAR 特征检索','Cross-Modal Change Attention');
 img(s,'CMCA.png',76,142,1125,367,'CMCA 结构及运算方向');
 defs(s,[['fb','成对的深层光学与 SAR 特征；下标 b 表示选定的交互层。'],['cb','该层通道数及空间高宽；两路输入均为 C_b × h_b × w_b。']],527,{rowH:61,keyW:260});
 note(s,'本文 Figure 2、III-D。核心方向是 Q 来自光学，K/V 来自 SAR。灾前光学提供结构上下文，灾后 SAR 提供待检索的状态信息。b 表示 CMCA 插入的选定语义层，在卷积模型中通常是瓶颈，在其他骨干中为对齐后的选定特征级。先给出模块位置，再展开 QKV、权重、检索和融合。');
}
{
 const s=ppt.slides.add();addHeader(s,'Q、K、V 构造：学习投影与 SAR 空间缩减','式（4）保持光学查询位置，减少被查询的 SAR 位置');
 eq(s,'eq4',76,175,550,266,'（4）');
 defs(s,[['qkv','Q 为查询；K 用于匹配；V 为实际被加权聚合的内容。'],['wqkv','三组可学习的 1×1 卷积，分别形成查询、键和值。'],['sr','核大小与步长均为 r 的深度卷积，仅作用于 SAR。'],['rm','把特征展开为序列，再按 M 个注意力头重排。']],157,{x:658,w:550,keyW:148,rowH:111,size:21});
 txt(s,'K 与 V 来自同一 SAR 特征，但使用不同投影；光学分支不经过 Sᵣ。',85,622,1110,42,23,C.navy,true);
 note(s,'本文式（4）。W_q、W_k、W_v 表示学习的线性通道投影，以 1×1 卷积实现。S_r 采用 kernel=stride=r 的 depth-wise 卷积来减少 SAR 空间 token 数，图中还包含归一化。R_M 只重排与分组，不负责几何配准。Q/K/V 是功能角色名称，而不是不同数据来源的三个原始图像。注意力头和每头维数在下一页定义。');
}
{
 const s=ppt.slides.add();addHeader(s,'图解式（4）：Q 来自光学，K 和 V 来自 SAR','UNet 示例：光学保留 1,600 个位置；SAR 经 r = 2 缩减至 400 个位置');
 txt(s,'光学深层特征',82,153,255,30,23,C.navy,true);
 txt(s,'512 × 40 × 40',82,183,255,27,18,C.muted);
 featureGrid(s,89,218,128,8,0,false);
 txt(s,'→',233,250,55,55,42,C.green,true);
 txt(s,'Wq',292,253,78,34,22,C.navy,true);
 txt(s,'→',365,250,55,55,42,C.green,true);
 for(let i=0;i<9;i++) rect(s,435+i*22,247,17,66,i%3===0?'#4E73AF':'#C8D8F0',C.white,0.5);
 txt(s,'Q：4 × 1600 × 128',680,261,404,35,23,C.navy,true);
 txt(s,'查询位置不缩减',681,301,360,27,18,C.ink);
 line(s,82,385,1114);
 txt(s,'SAR 深层特征',82,414,255,30,23,C.navy,true);
 txt(s,'512 × 40 × 40',82,444,255,27,18,C.muted);
 featureGrid(s,89,486,119,8,1,false);
 txt(s,'→',225,512,53,55,42,C.green,true);
 txt(s,'空间缩减 Sᵣ',282,496,192,31,21,C.navy,true);
 txt(s,'r = 2',317,535,100,29,18,C.muted);
 txt(s,'→',450,512,55,55,42,C.green,true);
 featureGrid(s,513,502,91,7,1,false);
 txt(s,'20 × 20',519,603,118,25,17,C.muted);
 txt(s,'→',624,512,53,55,42,C.green,true);
 txt(s,'Wk / Wv',680,478,173,34,22,C.navy,true);
 txt(s,'K、V：各 4 × 400 × 128',679,522,500,40,22,C.navy,true);
 txt(s,'同一 SAR 特征，经两组不同投影得到键和值。',679,571,503,57,18,C.ink);
 note(s,'图解对应论文式（4）与 Table IV 的 UNet 配置，彩色网格与短条仅为张量排列示意。光学特征通过 W_q 得 Q，空间位置数 N=40×40=1600。SAR 特征先经过核大小与步长均为 r=2 的空间缩减，得到 20×20=400 个位置，然后分别通过 W_k、W_v 形成 K、V。4 个头，每头 d=512/4=128；图中省略 batch 维。W_q/W_k/W_v 是可学习的 1×1 投影，R_M 进行多头重排。');
}
{
 const s=ppt.slides.add();addHeader(s,'张量维度：查询数、键值数与每头通道数','用明确的形状约束理解后续矩阵乘法');
 eq(s,'dims',83,151,1110,69);
 defs(s,[['md','M 是注意力头数；d 是每头维数；C_b 必须能被 M 整除。'],['n','N 是光学特征位置数，空间展开后成为查询序列长度。'],['ns','N_s 是缩减后 SAR 位置数，即 K/V 序列长度。'],['hs','h_s、w_s 为缩减后的 SAR 高宽；此处下标 s 标记缩减后的 SAR 网格。']],251,{rowH:66,keyW:280});
 txt(s,'UNet 示例：C_b = 512，h_b = w_b = 40，M = 4，r = 2',85,537,1110,36,23,C.navy,true);
 asset(s,'sym_unetdims.png',92,590,1090,53,'UNet QKV 张量尺寸');
 note(s,'本文式（4）后的维度定义与 Table IV。UNet 中 h_s=w_s=20，N=1600、N_s=400、d=128，所以 Q 为 4×1600×128，K/V 为 4×400×128。为简洁，均省略 batch 维。这里 h_s、w_s 是经 S_r 后的网格高宽，不应与输入 SAR 图像的 H、W 混用。空间缩减使查询关系数从 N² 降为 N×N_s；本例为原来的 1/4。');
}
{
 const s=ppt.slides.add();addHeader(s,'注意力权重：缩放点积与按 SAR 位置归一化','每个光学查询对全部缩减后的 SAR 位置计算检索权重');
 eq(s,'eq5',85,144,1095,115,'（5）');
 defs(s,[['kt','K 的最后两维转置；每头计算 (N × d) 与 (d × N_s) 的乘积。'],['sqrt','用每头维数 d 的平方根缩放点积，控制分数幅值。'],['softmax','沿 N_s 个 SAR 位置归一化；每个查询对应的权重非负且总和为 1。'],['ashape','A 是多头检索权重；每个头有 N 行，每行对应一个光学查询。']],302,{rowH:68,keyW:290});
 txt(s,'权重反映投影特征空间中的相关性，不等同于像素损伤概率。',85,624,1110,35,23,C.navy,true);
 note(s,'本文式（5）。单头 QK^T 的输出为 N×N_s，多头叠合为 M×N×N_s。softmax 的下标说明归一化轴，不是另一组参数。计算是跨模态检索：Q 来自光学，K 来自 SAR。每个光学位置可聚合多个 SAR 位置，不意味着注意力最大的位置就对应同一建筑。');
}
{
 const s=ppt.slides.add();addHeader(s,'SAR 特征检索：加权聚合并恢复光学位置布局','注意力矩阵作用于 V，而不是作用于原始 SAR 影像');
 eq(s,'eq6',85,153,1100,99,'（6）');
 defs(s,[['av','每头执行 (N × N_s)(N_s × d)，得到 N × d 的聚合表示。'],['rminv','合并 M 个头的通道，再将 N 个查询位置排回 h_b × w_b 网格。'],['retrieved','光学条件化的 SAR 特征；波浪号表示检索后的表示，上标 s 保留来源。']],299,{rowH:83,keyW:305});
 txt(s,'输出形状：C_b × h_b × w_b，与光学查询网格一致。',85,577,1110,39,25,C.navy,true);
 txt(s,'该输出是特征表示，不是通过几何变换得到的配准 SAR 图像。',85,629,1110,31,22,C.muted);
 note(s,'本文式（6）。A 的列与 V 的位置维 N_s 对齐，因此矩阵乘法输出保留光学位置数 N。R_M^{-1} 是重排操作的逆过程，恢复通道与二维布局；不对注意力权重做逆矩阵运算。此处波浪号只表示被检索的中间特征，后文 DA-CutMix 的波浪号表示混合样本，含义由上下文区分。');
}
{
 const s=ppt.slides.add();addHeader(s,'图解式（5）–（6）：光学位置如何检索 SAR 信息','矩阵仅画出少量行列作为示意；实际 UNet 每头有 1,600 × 400 个权重');
 txt(s,'光学查询 Q',76,174,176,31,21,C.navy,true);
 txt(s,'N × d',85,207,160,27,18,C.muted);
 for(let r=0;r<7;r++) for(let c=0;c<4;c++) rect(s,86+c*29,259+r*24,25,20,(r===3?'#4E73AF':'#C8D8F0'),C.white,0.3);
 rect(s,84,331,116,26,'none',C.gold,3);
 txt(s,'×',208,315,45,49,37,C.green,true);
 txt(s,'SAR 键 Kᵀ',261,174,180,31,21,C.navy,true);
 txt(s,'d × Nₛ',267,207,160,27,18,C.muted);
 for(let r=0;r<4;r++) for(let c=0;c<7;c++) rect(s,265+c*24,291+r*29,20,25,(c%3===0?'#279A72':'#BDE9D8'),C.white,0.3);
 txt(s,'→',441,316,45,50,38,C.green,true);
 txt(s,'注意力权重 A',494,174,224,31,21,C.navy,true);
 txt(s,'N × Nₛ',503,207,179,27,18,C.muted);
 for(let r=0;r<7;r++) for(let c=0;c<7;c++){
  const v=(r*3+c*5)%4;
  rect(s,499+c*24,267+r*24,20,20,['#EAF6F3','#BDE9D8','#72C5A5','#279A72'][v],C.white,0.3);
 }
 rect(s,497,338,170,25,'none',C.gold,3);
 txt(s,'softmax：每行权重和为 1',485,449,290,32,19,C.ink);
 txt(s,'×',682,317,42,50,37,C.green,true);
 txt(s,'SAR 值 V',724,174,180,31,21,C.navy,true);
 txt(s,'Nₛ × d',730,207,160,27,18,C.muted);
 for(let r=0;r<7;r++) for(let c=0;c<4;c++) rect(s,732+c*28,259+r*24,24,20,(r%3===0?'#805AA7':'#DEC9ED'),C.white,0.3);
 txt(s,'→',858,317,48,50,38,C.green,true);
 txt(s,'检索后特征',919,174,250,31,21,C.navy,true);
 txt(s,'N × d',932,207,167,27,18,C.muted);
 featureGrid(s,939,269,154,7,2,false);
 txt(s,'重排回光学位置网格',916,450,286,34,18,C.ink);
 line(s,82,514,1113);
 txt(s,'第 i 个光学位置：比较所有 SAR 键，得到 A 的第 i 行；再用这行权重加权求和所有 SAR 值。',83,541,1110,69,21,C.navy,true);
 note(s,'图解对应论文式（5）与式（6），矩阵色块只表示运算结构而非实测权重。对每一个注意力头，Q 的形状为 N×d，K 转置为 d×N_s，QK^T 为 N×N_s。softmax 沿 N_s 即每一行的 SAR 位置归一化，权重非负且每行和为 1。A 与 V(N_s×d) 相乘得到 N×d，之后合并 M 个头并重排回 h_b×w_b 的光学位置网格。图中金色轮廓对应某个光学查询及其在 A 中的一整行。');
}
{
 const s=ppt.slides.add();addHeader(s,'变化特征：学习灾前结构与灾后响应的组合','损伤监督决定哪些跨模态组合有助于最终分类');
 eq(s,'eq7',85,149,1100,98,'（7）');
 defs(s,[['concat','将检索后的 SAR 表示与原光学特征沿通道拼接，得到 2C_b 通道。'],['phi','1×1 卷积、批归一化与 ReLU 激活，学习跨模态信息组合并映射回 C_b。'],['fc','CMCA 输出；上标 c 表示变化特征，下标 b 与输入的交互层一致。']],291,{rowH:80,keyW:245});
 asset(s,'sym_phi_dims.png',92,552,1090,49,'变化融合的输入输出张量尺寸');
 txt(s,'学习融合避免把不同成像机制下的特征关系固定为逐元素相减。',85,622,1110,41,23,C.navy,true);
 note(s,'本文式（7）。输入两组表示都有 C_b×h_b×w_b，拼接后为 2C_b×h_b×w_b，φ 输出 C_b×h_b×w_b。输出 c 是 learned change-aware representation，不是直接的变化概率图。损伤分类损失通过反向传播共同学习编码器、Q/K/V 投影和 φ。');
}
{
 const s=ppt.slides.add();addHeader(s,'输出适配：拼接投影保持原解码器接口','UNet、DeepLabV3+ 与 DamageFormer 使用该形式');
 eq(s,'eq8',85,153,1100,94,'（8）');
 defs(s,[['fout','适配后的输出特征；上标 out 表示送入原解码器或预测头的接口。'],['psi','可学习的通道投影，将三组 C_b 通道映射为下游需要的通道数。'],['concat','拼接原光学、原 SAR、CMCA 变化特征，总计 3C_b 通道。']],287,{rowH:83,keyW:245});
 asset(s,'sym_psi_dims.png',92,553,1090,50,'拼接适配的输入输出尺寸');
 txt(s,'这里 C_dec 表示下游接口通道数；空间高宽仍为 h_b × w_b。',85,624,1110,35,22,C.navy,true);
 note(s,'本文式（8）。C_dec 是本页为说明维度引入的记号，表示原解码器或预测头实际接收的通道数，具体取值随骨干而定；并非论文新增超参数。ψ 与 φ 的职责不同：φ 学习变化表示，ψ 将输出接入原网络。本形式保留原光学和 SAR 表征，同时加入 CMCA 表征。');
}
{
 const s=ppt.slides.add();addHeader(s,'输出适配：直接替换与残差注入','三种适配器只改变连接方式，CMCA 核心注意力保持一致');
 txt(s,'直接替换：SiamAttnUNet',83,151,1110,35,25,C.navy,true);
 eq(s,'replace',88,199,1095,70);
 txt(s,'当原接口接收 C_b 通道时，直接使用 CMCA 输出。',87,285,1100,38,23);
 line(s,80,343,1120);
 txt(s,'残差注入：SiamCRNN、ChangeMamba',83,367,1110,35,25,C.red,true);
 eq(s,'residual',88,413,1095,80);
 defs(s,[['fbase','原骨干已经融合的特征；上标 base 表示原有融合路径。'],['rho','将变化特征投影到与原融合特征相同的通道数，再逐元素相加。']],523,{rowH:61,keyW:260});
 note(s,'本文式（8）后的适配说明。替换要求 CMCA 输出通道与原接口相容。残差相加要求两个张量的通道数和空间尺寸均相同，ρ 负责通道匹配，空间网格在所选交互层已匹配。out、base、c 分别表示适配输出、原融合特征、变化特征。具体插入层与配置仍保留在附录。');
}
{
 const s=ppt.slides.add();addHeader(s,'DA-CutMix：跨事件损伤区域的同步增强','目标是增加稀有损伤类别的训练曝光，并保留区域内部对应');
 img(s,'DA-CutMix.png',73,152,730,494,'DA-CutMix 训练算法流程');
 txt(s,'跨事件供体',843,174,365,37,25,C.navy,true);
 txt(s,'供体来自不同训练事件，\n增加损伤与场景组合。',843,229,365,91,23);
 txt(s,'损伤含量约束',843,347,365,37,25,C.green,true);
 txt(s,'同时满足绝对像素数\n和相对占比门槛。',843,401,365,91,23);
 txt(s,'双模态与标签同步替换',843,520,365,71,24,C.red,true);
 txt(s,'训练启用，验证和推理关闭。',843,600,365,59,22);
 note(s,'本文 III-E、Figure 3。DA-CutMix 的新增约束是供体事件不同，以及区域中受损/毁坏像素满足显式门槛。它扩展训练分布，不声称生成物理上完整的一次灾害场景。插入区域内部的光学、SAR、标签必须对应同一供体位置。下一页开始定义图中的 T_i、T_j、e_i、e_j 等符号。');
}
{
 const s=ppt.slides.add();addHeader(s,'训练样本与事件约束：区分目标、供体及标签','式（9）表示一个配准训练样本，事件不等条件约束供体来源');
 eq(s,'eq9',85,150,1100,88,'（9）');
 defs(s,[['ti','目标样本四元组，包含两种影像、损伤标签和事件标识。'],['ij','i 为目标样本索引，j 为供体样本索引，均来自训练集。'],['xi','两张三通道 H × W 影像，以及一张 H × W 的四类标签图。'],['ei','灾害事件标识；要求 e_j ≠ e_i，排除来自同一事件的供体。'],['p','是否尝试混合的伯努利概率；0.25 / 0.30 由验证集选择。']],278,{rowH:66,keyW:265});
 txt(s,'目标与供体各自执行同步几何增强；每份样本内部的三层数据共享变换。',85,629,1110,35,22,C.navy,true);
 note(s,'本文式（9）、III-E、Table III。样本索引 i/j 与编码阶段 l、交互层 b 属于不同索引，不应混用。Y_i 是离散标签，不是模型输出；e_i 是事件编号，不参与网络特征计算。先对目标施加同步 crop/flip/rotation，再以概率 p 尝试 CutMix；供体有自己的一套同步几何参数，不要求目标与供体用相同参数。');
}
{
 const s=ppt.slides.add();addHeader(s,'区域接受准则：损伤计数与比例双约束','式（10）以供体标签筛选矩形区域，而非按影像亮度筛选');
 eq(s,'eq10',84,146,1105,105,'（10）');
 defs(s,[['bu','B 为候选矩形，u 为其中一个像素坐标；求和遍历框内所有像素。'],['yb','供体 j 在坐标 u 处的真实标签。'],['cd','目标类别集合：2 为受损，3 为毁坏。'],['indicator','指示函数：括号内条件成立取 1，否则取 0；求和即目标像素总数。'],['bsize','候选框总像素数，等于框高乘以框宽。'],['threshold','n_min 为最低数量，τ 为最低占比；max 同时保证两项条件。']],280,{rowH:61,keyW:290,size:21});
 note(s,'本文式（10）。数学求和符号 Σ 表示对每一个 u∈B 累加，∈ 表示集合成员关系，≥ 表示接受区域至少达到门槛。左侧只统计标签属于 C_d 的像素。max(n_min,τ|B|) 的等价条件是目标像素数至少 n_min 且占比至少 τ。区域中可同时保留背景和完好像素作为上下文。');
}
{
 const s=ppt.slides.add();addHeader(s,'区域采样：门槛计算与拒绝采样过程','计算示例与论文实现设置');
 table(s,[['候选区域','总像素 |B|','τ|B|，τ = 0.05','最终最低损伤像素数'],['80 × 80','6,400','320','max(200, 320) = 320'],['160 × 160','25,600','1,280','max(200, 1,280) = 1,280']],75,170,1130,236,[240,230,280,380],22);
 txt(s,'80 × 80 区域中，250 个损伤像素满足绝对门槛，但未达到 5% 的比例要求。',84,446,1110,66,24,C.navy,true);
 txt(s,'框高和框宽各自按输入裁剪尺寸的 12%–35% 采样。',84,534,1110,40,23);
 txt(s,'最多尝试 10 个供体，每个供体最多 10 个候选框；失败时返回原目标样本。',84,593,1110,62,23);
 note(s,'本文 Table III、IV-D，式（10）的数值代入。示例不是实测结果。n_min=200，τ=0.05，输入裁剪为 640×640。概率 p 决定是否尝试增强，不是损伤比例；τ 决定区域的损伤占比，不是增强概率。候选区域被拒绝后继续采样，在预设次数内找不到可用区域则保持原样本。');
}
{
 const s=ppt.slides.add();addHeader(s,'同步混合：共享二值掩膜替换三层数据','式（11）保持供体区域内部的光学、SAR 与标签一致');
 eq(s,'eq11',84,143,1105,242,'（11）');
 defs(s,[['mb','二值空间掩膜，接受框 B 内为 1、框外为 0；后者由 1−M_B 选择。'],['odot','逐元素乘法；H × W 掩膜在影像的三个通道上广播。'],['tilde','波浪号表示混合结果；下标 j 取供体内容，下标 i 保留目标内容。']],417,{rowH:67,keyW:300,size:21});
 txt(s,'标签做空间替换，不做类别概率或面积比例的插值；输出形状与输入相同。',85,630,1110,35,22,C.navy,true);
 note(s,'本文式（11）。M_B 与 1−M_B 是互补选择掩膜；⊙ 在每个位置相乘，加号合并两块互补区域。因为掩膜只有 0/1，标签仍取 0/1/2/3，不生成混合类别。两张影像和标签必须使用同一个 B。此处波浪号表示增强后的样本，与式（6）中“检索后的特征”含义不同，不能把两处波浪号理解为同一种运算。');
}
{
 const s=ppt.slides.add();addHeader(s,'图解式（10）–（11）：先筛选，再同步替换','计算数字为门槛示例；下方三层示意同一个二值掩膜 M_B');
 txt(s,'供体标签中的候选框 B',84,155,340,32,22,C.navy,true);
 rect(s,92,209,170,132,C.white,C.navy,1.4);
 for(let r=0;r<7;r++) for(let c=0;c<9;c++){
  const fill=(r+c*2)%7<2?'#BD4748':((r*3+c)%5<2?'#EAC08D':'#FFFFFF');
  rect(s,97+c*18,214+r*18,16,16,fill,C.white,0.3);
 }
 rect(s,125,233,81,74,'none',C.gold,3);
 txt(s,'80 × 80 候选区域',292,205,279,34,21,C.ink,true);
 txt(s,'|B| = 6,400',292,250,260,31,19,C.ink);
 txt(s,'最低数量 = max(200, 0.05 × 6,400) = 320',580,205,585,64,21,C.navy,true);
 txt(s,'示例含 400 个受损或毁坏像素：400 ≥ 320，接受 B。',580,276,589,54,20,C.green,true);
 line(s,82,377,1113);
 txt(s,'通过筛选后，同一个 B 用于光学、SAR 和标签',83,393,1110,35,23,C.navy,true);
 txt(s,'目标样本 i',184,430,215,28,18,C.muted);
 txt(s,'供体样本 j',474,430,215,28,18,C.muted);
 txt(s,'混合样本',799,430,215,28,18,C.muted);
 for(const [row,label,base,patch] of [
  [0,'光学','#C8D8F0','#4E73AF'],
  [1,'SAR','#D7DEE3','#647480'],
  [2,'标签','#EAF6F3','#BD4748']
 ]){
  const y=470+row*54;
  txt(s,label,83,y+5,79,29,19,C.ink,true);
  rect(s,176,y,206,34,base,C.line,0.8);
  rect(s,237,y,58,34,'none',C.gold,2.5);
  txt(s,'+',414,y-2,43,41,30,C.green,true);
  rect(s,466,y,206,34,base,C.line,0.8);
  rect(s,527,y,58,34,patch,C.gold,2.5);
  txt(s,'→',713,y-4,58,43,35,C.green,true);
  rect(s,792,y,206,34,base,C.line,0.8);
  rect(s,853,y,58,34,patch,C.gold,2.5);
 }
 txt(s,'黄色框在三层中的位置完全相同，标签仍保持离散类别。',83,647,1100,29,18,C.ink);
 note(s,'图解对应论文式（10）与式（11）。上方 80×80、400 个损伤像素是计算示例，不是实测区域；n_min=200、τ=0.05 均来自论文设置。通过条件为目标类别像素数 ≥ max(n_min,τ|B|)，400≥320 时接受。下方条带仅示意：同一个二值掩膜 M_B 对目标与供体的光学影像、SAR 影像和标签执行相同空间替换。黄色框内取供体 j，框外保留目标 i。标签是离散替换，不做数值插值。供体与目标须来自不同事件；没有通过筛选时不执行替换。');
}
{
 const s=ppt.slides.add();addHeader(s,'DA-CutMix 样例：跨事件内容与标签同步插入','增强只改变训练输入，不修改骨干结构或推理流程');
 img(s,'dacutmix_example.png',75,184,1130,375,'双模态与标签同步替换的样本');
 txt(s,'黄色框在灾前光学、灾后 SAR 和标签图中使用同一坐标区域。',83,591,1110,52,25,C.navy,true);
 note(s,'本文 Figure 4。三幅图依次为灾前光学、灾后 SAR、类别标签。此图检查的是区域内部对齐和标签同步，不能据此单独证明泛化性能。来自不同事件的内容增加训练组合多样性，具体收益需由消融与逐事件实验判断。');
}
{
 const s=ppt.slides.add();addHeader(s,'损伤分割目标：像素监督与区域重叠优化','单头模型优化四类损伤图；标签取值为 0、1、2、3');
 eq(s,'eq12',85,153,1100,89,'（12）');
 defs(s,[['ldam','损伤分割损失；下标 dam 标记四类损伤任务。'],['ce','多类交叉熵，按像素约束正确类别的预测概率。'],['lovasz','Lovasz-Softmax，优化类别 IoU 的可微替代目标。'],['lamdam','Lovasz 项的权重，控制它相对交叉熵的贡献。']],287,{rowH:72,keyW:270});
 txt(s,'IoU 为预测区域与真实区域的交集 / 并集；CE 与 Lovasz 从不同角度约束预测。',85,621,1110,47,22,C.navy,true);
 note(s,'本文式（12）、III-F。L 表示优化目标而不是模型输出，CE 是 cross-entropy 的缩写。Lovasz-Softmax 提供针对类别 IoU 的可微替代目标，并不意味着其数值等于 1−IoU。UNet、SiamAttnUNet、DeepLabV3+ 沿用单损伤头结构。本方法没有额外设置 CMCA 专用损失。');
}
{
 const s=ppt.slides.add();addHeader(s,'双头训练目标：增加二类建筑定位监督','DamageFormer、SiamCRNN 与 ChangeMamba 沿用辅助定位头');
 eq(s,'eq13',85,150,1100,96,'（13）');
 defs(s,[['ltotal','总目标：四类损伤损失，加上二类建筑定位损失。'],['loc','上标 loc 表示建筑定位任务；建筑三类合并为前景，与背景二分类。'],['locloss','定位分支的交叉熵与 Lovasz；公式中的 L_dam 由式（12）定义。'],['lamloc','定位 Lovasz 项的权重；定位交叉熵前的系数为 1。']],287,{rowH:73,keyW:300});
 txt(s,'CMCA 与 DA-CutMix 沿用原骨干的输出结构及监督方式，未引入额外损失项。',85,622,1110,43,22,C.navy,true);
 note(s,'本文式（13）。单头只输出损伤图，双头额外输出建筑/背景图。L_total 与 L_dam 不可混用；L_dam 已包含 λ_dam=0.75 的四类 Lovasz，而 L_total 另外加入 λ_loc=0.5 的定位 Lovasz。辅助定位由原损伤标签派生监督，不要求新增人工标签。');
}
{
 const s=ppt.slides.add();addHeader(s,'图解式（12）–（13）：两个预测头分别接受什么监督','定位头仅适用于原骨干本来就有辅助建筑定位输出的情况');
 txt(s,'共享特征',78,187,194,32,23,C.navy,true);
 featureGrid(s,89,278,145,7,0,false);
 rect(s,243,348,70,3,C.green);
 rect(s,311,259,3,186,C.green);
 txt(s,'→',311,238,57,53,40,C.green,true);
 txt(s,'→',311,420,57,53,40,C.green,true);
 txt(s,'损伤头',387,193,180,33,22,C.navy,true);
 txt(s,'四类像素图',387,227,197,30,18,C.ink);
 for(const [i,fill] of ['#FFFFFF','#41BD82','#EAC08D','#BD4748'].entries())
  rect(s,389+i*42,283,32,57,fill,C.line,0.8);
 txt(s,'背景   完好   受损   毁坏',387,356,326,29,17,C.ink);
 txt(s,'→',719,277,50,54,38,C.green,true);
 txt(s,'Ldam = CE + λdam × Lovasz',777,249,424,77,22,C.navy,true);
 txt(s,'逐像素分类 + 类别区域重叠',776,331,426,43,18,C.ink);
 line(s,382,402,811);
 txt(s,'定位头（双头骨干）',387,422,291,33,22,C.navy,true);
 txt(s,'建筑 / 背景二类图',387,457,287,29,18,C.ink);
 rect(s,390,507,81,54,C.white,C.line,0.8);
 rect(s,473,507,81,54,C.green,C.line,0.8);
 txt(s,'背景',393,573,74,29,17,C.ink);
 txt(s,'建筑',476,573,74,29,17,C.ink);
 txt(s,'→',719,502,50,54,38,C.green,true);
 txt(s,'Lloc = CEloc + λloc × Lovaszloc',777,478,425,89,21,C.navy,true);
 txt(s,'建筑定位只提供辅助监督',777,567,422,36,18,C.ink);
 txt(s,'单头：只用 Ldam。双头：Ltotal = Ldam + Lloc；不增加 CMCA 专用损失。',83,629,1110,40,20,C.ink,true);
 note(s,'图解对应论文式（12）和式（13），网格与色块只示意任务输出。损伤头输出背景、完好、受损、毁坏四类，由多类交叉熵和 Lovasz-Softmax 联合监督。仅原本具有辅助建筑定位头的 DamageFormer、SiamCRNN、ChangeMamba 另输出建筑/背景二类图，使用定位交叉熵及其 Lovasz 项。符号 Lloc 在本图中是两个定位损失项的合称，不是论文新增独立公式；总损失等价于论文式（13）。λdam=0.75，λloc=0.5。CMCA 与 DA-CutMix 不增加额外损失项。');
}
{
 const s=ppt.slides.add();addHeader(s,'方法归纳：特征交互与训练分布两条改进路径','保持模块职责清晰，便于独立消融');
 table(s,[['模块','作用对象','关键约束与计算','推理时'],['CMCA','成对深层特征','光学 Q 查询 SAR K/V；\n加权检索、学习融合、接口适配','保留'],['DA-CutMix','训练影像与标签','跨事件供体、损伤数量/比例约束；\n三层数据共享掩膜','关闭']],75,176,1130,296,[210,235,510,175],22);
 txt(s,'CMCA 检验跨模态交互的收益；DA-CutMix 检验损伤样本曝光的收益。',85,522,1110,67,25,C.navy,true);
 txt(s,'后续通过原骨干、分别启用、联合启用四种配置，区分独立贡献与组合效果。',85,611,1110,53,23);
 note(s,'本文 III 与 IV-E。两项方法解决不同环节的问题，既可以独立启用，也可以组合。跨事件增强旨在增加训练多样性，不直接等价于已经证明未见事件泛化。后续实验展示标准监督测试集、逐类与逐事件的表现。详细骨干配置保留在附录。');
}

// 12 divider
addSection('03','实验结果与分析','Six backbones on the BRIGHT benchmark',12);
// 13 protocol
{
 const s=ppt.slides.add();addHeader(s,'实验设置','Standard supervised split and common training protocol',13);
 const values=[
  ['设置','取值','说明'],
  ['输入裁剪','640 × 640','双模态与标签同步几何增强'],
  ['训练','50,000 iterations; batch 16','AdamW, lr = 1×10⁻⁴'],
  ['DA-CutMix 概率','0.25 或 0.30','使用验证集选择'],
  ['主要指标','mIoU','四类 IoU 的平均'],
  ['辅助指标','F1loc / F1clf / OA','定位、损伤分类和总体精度']
 ];
 table(s,values,72,170,1135,418,[265,350,520],18);
 txt(s,'所有对比均使用相同数据划分与优化协议。',77,615,1100,32,20,C.navy,true);
 note(s,'论文 Table III 与第 IV 节。测试集最终报告，验证集选模型与 CutMix 概率。F1clf 是建筑三类 F1 的调和平均，OA 易受背景主导。');
}
// 14 ablation design
{
 const s=ppt.slides.add();addHeader(s,'消融设计：两个独立开关','Baseline, CMCA, DA-CutMix, and their combination',14);
 const vals=[
  ['配置','CMCA','DA-CutMix','测试目的'],
  ['Baseline','关闭','关闭','原始骨干'],
  ['+ CMCA','开启','关闭','模型级特征交互'],
  ['+ DA-CutMix','关闭','开启','数据级稀有类增强'],
  ['+ CMCA + DA-CutMix','开启','开启','两者的组合效果']
 ];
 table(s,vals,78,181,1124,353,[350,160,210,404],20,[4]);
 txt(s,'每个骨干分别运行四种配置，共 24 组测试结果。',83,573,1100,44,23,C.ink,true);
 note(s,'论文 Table V 含六个骨干的四种配置，CutMix 概率从 0.25 与 0.30 中按验证集选择。');
}
// 15 mIoU table
{
 const s=ppt.slides.add();addHeader(s,'标准测试集：六骨干 mIoU 消融','Values in percent; Δ compares the combination with baseline',15);
 const vals=[
  ['骨干','Baseline','+ CMCA','+ DA-CutMix','组合','组合增量'],
  ['UNet','64.74','65.39','64.46','66.16','+1.42'],
  ['DamageFormer','66.84','66.55','67.20','67.14','+0.30'],
  ['SiamAttnUNet','65.03','65.05','65.28','66.28','+1.25'],
  ['SiamCRNN','66.13','66.25','66.36','66.52','+0.39'],
  ['DeepLabV3+','63.85','64.83','65.25','65.79','+1.94'],
  ['ChangeMamba','66.73','67.32','67.04','67.43','+0.70']
 ];
 table(s,vals,62,171,1152,425,[250,170,180,195,170,187],18,[2,6]);
 txt(s,'组合方案在 5/6 个骨干组内取得最高 mIoU；DamageFormer 单独使用 DA-CutMix 更高。',71,617,1135,41,19,C.ink,true);
 note(s,'论文 Table V。组合相对六个原始骨干基线的 mIoU 增益为 +0.30 至 +1.94 个百分点。DamageFormer 组 +DA-CutMix=67.20，组合=67.14，需保留这个例外。');
}
// 16 class-wise
{
 const s=ppt.slides.add();addHeader(s,'损伤类别：增益存在骨干差异','Per-class IoU, baseline vs combined configuration',16);
 const vals=[
  ['骨干','Damaged 基线','Damaged 组合','变化','Destroyed 基线','Destroyed 组合','变化'],
  ['UNet','37.48','39.08','+1.60','53.59','55.49','+1.90'],
  ['DamageFormer','39.97','40.17','+0.20','56.06','57.12','+1.06'],
  ['SiamAttnUNet','38.24','38.94','+0.70','53.60','57.30','+3.70'],
  ['SiamCRNN','40.22','40.10','−0.12','54.17','56.06','+1.89'],
  ['DeepLabV3+','37.06','39.26','+2.20','51.45','54.12','+2.67'],
  ['ChangeMamba','40.74','41.82','+1.08','56.09','57.25','+1.16']
 ];
 table(s,vals,53,177,1174,414,[200,163,163,117,166,169,196],16,[4]);
 txt(s,'SiamCRNN 的 Damaged IoU 略降，说明总 mIoU 提升不等于每一类同步提升。',68,615,1140,42,19,C.ink,true);
 note(s,'论文 Table V 逐类 IoU。SiamCRNN 组合配置 Damaged 为 40.10，基线 40.22，下降 0.12 个百分点。');
}
// 17 per-event
{
 const s=ppt.slides.add();addHeader(s,'逐事件评估：全局结果之外的差异','Unweighted mean of 14 event-wise mIoU values',17);
 const vals=[
  ['骨干 / 报告配置','基线均值','配置均值','变化','提升事件'],
  ['UNet / 组合','48.09','51.59','+3.50','12/14'],
  ['DamageFormer / DA-CutMix','52.04','51.68','−0.36','7/14'],
  ['SiamAttnUNet / 组合','48.49','49.85','+1.36','11/14'],
  ['SiamCRNN / 组合','50.29','51.11','+0.82','7/14'],
  ['DeepLabV3+ / 组合','46.79','49.08','+2.29','12/14'],
  ['ChangeMamba / 组合','50.39','51.82','+1.43','10/14']
 ];
 table(s,vals,65,178,1150,404,[430,170,170,175,205],17,[2]);
 txt(s,'DamageFormer 的全局 mIoU 最优配置，在事件平均值上反而下降 0.36 个百分点。',70,606,1135,52,19,C.red,true);
 note(s,'论文 Table VI。这里是 14 个事件 mIoU 的不加权平均；它与按所有测试像素汇总的全局 mIoU 不同。DamageFormer 使用其组内全局最佳的 +DA-CutMix 配置。');
}
// 18 qualitative 1
{
 const s=ppt.slides.add();addHeader(s,'可视化结果：爆炸、火山、野火','Qualitative comparison across the six combined configurations',18);
 img(s,'qualitative_results.png',86,154,1100,469,'定性结果上半部分：爆炸、火山、野火', 'cover', {left:0,top:0,right:0,bottom:0.57});
 txt(s,'主要建筑群及集中毁坏区域得到恢复；局部受损与完好仍会混淆。',91,634,1100,35,19,C.ink,true);
 note(s,'论文 Fig. 5 上三行。爆炸场景中，一些模型找到建筑却将受损建筑标成完好；野火场景中上缘少量受损建筑识别不稳定。');
}
// 19 qualitative 2
{
 const s=ppt.slides.add();addHeader(s,'可视化结果：洪水、飓风、地震、冲突','Damage severity remains difficult near class boundaries',19);
 img(s,'qualitative_results.png',86,154,1100,469,'定性结果下半部分：洪水、飓风、地震、冲突','cover',{left:0,top:0.43,right:0,bottom:0});
 txt(s,'洪水场景中，部分毁坏建筑被判为受损；少量孤立受损建筑容易漏判。',91,634,1100,35,19,C.ink,true);
 note(s,'论文 Fig. 5 下四行。洪水场景中 DeepLabV3+ 将部分毁坏区域判为受损。地震场景中少量受损建筑嵌于完好建筑群，识别不稳定。');
}
// 20 limitations
{
 const s=ppt.slides.add();addHeader(s,'当前局限','Observed failure cases and scope of evidence',20);
 label(s,'01','Damaged 类仍难','局部损伤证据弱，模型能找到建筑边界，却可能误判损伤等级。',90,173,1080,C.red);
 label(s,'02','跨事件收益不均','DamageFormer 的逐事件均值下降；部分骨干的单类 IoU 也存在回落。',90,318,1080,C.gold);
 label(s,'03','泛化验证仍需补充','当前实验使用 BRIGHT 标准监督划分，尚未覆盖留一事件、留一灾种及跨数据集测试。',90,463,1080,C.green);
 note(s,'论文讨论与结论。不要将标准监督测试集结果解释为已证明零样本跨事件泛化。');
}
// 21 conclusion and future
{
 const s=ppt.slides.add();addHeader(s,'总结与下一步','Conclusion and next research steps',21);
 txt(s,'当前结论',78,166,500,35,23,C.green,true);
 txt(s,'CMCA 提供有方向的光学—SAR 特征交互。',80,213,1090,39,24,C.ink,true);
 txt(s,'DA-CutMix 在训练中增加跨事件损伤样本，推理时无额外计算。',80,264,1090,64,24,C.ink,true);
 txt(s,'组合方案在六骨干基线上均提升 mIoU，最佳达到 67.43%。',80,341,1090,45,24,C.ink,true);
 line(s,79,412,1110);
 txt(s,'下一步',78,442,500,35,23,C.red,true);
 txt(s,'按类别与难度自适应选择供体事件、区域大小和增强概率。',80,486,1090,39,21,C.ink);
 txt(s,'增加留一事件、留一灾种与跨数据集评估。',80,535,1090,39,21,C.ink);
 txt(s,'融合多尺度与 SAR 散射信息，降低注意力计算量。',80,584,1090,39,21,C.ink);
 note(s,'论文结论。67.43% 为 ChangeMamba + CMCA + DA-CutMix 在 BRIGHT 标准测试集上的 mIoU。未来工作按论文结论列出。');
}
addSection('04','附录：骨干接入与配置','Backbone integration and attention configuration');
// 9 CMCA integration
{
 const s=ppt.slides.add();addHeader(s,'附录：CMCA 在六种骨干中的接入位置','One two-input / one-output feature interface',9);
 const rows=[
  ['骨干','插入特征','输出适配'],
  ['UNet / DeepLabV3+','最深卷积特征','拼接后投影'],
  ['SiamAttnUNet','第 5 个编码阶段','替换原融合特征'],
  ['DamageFormer','融合后的 Swin 金字塔特征','拼接后投影'],
  ['SiamCRNN / ChangeMamba','最深语义特征','残差注入']
 ];
 table(s,rows,70,185,1140,330,[280,505,355],19);
 txt(s,'六个骨干共享 CMCA 核心计算，仅调整通道数、注意力头数、降采样倍率和输出适配器。',76,559,1110,60,21,C.ink,true);
 note(s,'论文 Table IV 给出骨干特定的插入位置、通道数、头数、降采样倍率。CMCA 的注意力核心不变，原解码器或预测头保留。');
}
{
 const s=ppt.slides.add();addHeader(s,'附录：CMCA 配置与张量尺寸示例','Backbone-specific configuration for a 640 × 640 input crop');
 table(s,[
  ['骨干','通道 C_b','特征 h_b × w_b','头数 M','降采样 r','Nₛ'],
  ['UNet','512','40 × 40','4','2','400'],
  ['SiamAttnUNet','1024','40 × 40','8','2','400'],
  ['DamageFormer','256','160 × 160','4','16','100'],
  ['SiamCRNN','512','20 × 20','8','1','400'],
  ['DeepLabV3+','2048','40 × 40','8','2','400'],
  ['ChangeMamba','768','20 × 20','8','1','400']
 ],71,166,1138,349,[280,160,230,160,160,148],19,[1]);
 txt(s,'以 UNet 为例',80,548,1000,32,23,C.green,true);
 txt(s,'N = 1,600，Nₛ = 400，d = 128。每个头的注意力矩阵为 1,600 × 400。',80,591,1100,55,23,C.ink,true);
 note(s,'论文 Table IV。N_s 根据表内特征尺寸与空间缩减倍率计算。UNet 中 Q 为 4×1600×128，K/V 为 4×400×128，A 为 4×1600×400。该页的尺寸示例是公式的直接代入，不是新增实验。');
}


// 22 Q&A
{
 const s=ppt.slides.add();s.background.fill=C.dark;
 txt(s,'THANK YOU',64,222,700,30,18,'#B8C4CC',true);
 txt(s,'谢谢大家 · Q & A',64,283,1100,84,54,C.white,true);
 rect(s,0,415,W,2,C.lime);
 txt(s,'陈昊  ·  2026年10月',64,601,700,32,19,C.white);
}

await fs.mkdir(buildDir,{recursive:true});
await fs.mkdir(outputDir,{recursive:true});
const draftPath=path.join(buildDir,'group_meeting_draft.pptx');
await (await PresentationFile.exportPptx(ppt)).save(draftPath);
const dedupPath=path.join(buildDir,'group_meeting_dedup.pptx');
console.log(execFileSync(py,[path.join(buildDir,'deduplicate_media.py'),draftPath,dedupPath],{encoding:'utf8'}));
const fontPath=path.join(buildDir,'candidate.pptx');
execFileSync(py,[path.join(buildDir,'patch_fonts.py'),dedupPath,fontPath]);
const mergedPath=path.join(buildDir,'candidate_with_current_edits.pptx');
execFileSync(py,[path.join(buildDir,'transplant_current.py'),path.join(outputDir,'组会汇报_202610_陈昊_CMCA_DACutMix_技术讲解与符号详释版.pptx'),fontPath,mergedPath]);
const { finalizePresentation }=await import(pathToFileURL(path.join(skillDir,'container_tools/artifact_tool_utils.mjs')).href);
const result=await finalizePresentation({
  workspaceDir,candidatePath:mergedPath,finalPath,pythonExecutable:py,
  integrityValidatorPath:path.join(skillDir,'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath:path.join(skillDir,'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit',
    ...nativeTableSlides.flatMap(n=>['--require-native-table-slide',String(n)] )],
  requiredNativeTableOwnerSlides:nativeTableSlides,
  fontPolicy:{basis:'user_request',families:['Microsoft YaHei','Times New Roman'],scriptFonts:{ea:'Microsoft YaHei'}},
  verifyArtifactToolImport:true,
  receiptPath:path.join(buildDir,'validation.json')
});
console.log(JSON.stringify({finalPath,result},null,2));
