import path from 'node:path';
import { pathToFileURL } from 'node:url';

const workspaceDir='/Users/haochen/Documents/Development/bright-cmca';
const buildDir=path.join(workspaceDir,'.codex-build-activation-clarity');
const skillDir='/Users/haochen/.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11127/skills/presentations';
const {finalizePresentation}=await import(pathToFileURL(path.join(skillDir,'container_tools/artifact_tool_utils.mjs')).href);
const result=await finalizePresentation({
 workspaceDir,
 candidatePath:path.join(buildDir,'candidate.pptx'),
 finalPath:path.join(workspaceDir,'output/pptx/组会汇报_202610_陈昊_CMCA_DACutMix_方法公式图解_示意说明修订版.pptx'),
 pythonExecutable:'/Users/haochen/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3',
 integrityValidatorPath:path.join(skillDir,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skillDir,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit'],
 fontPolicy:{basis:'user_request',families:['Microsoft YaHei','Times New Roman'],scriptFonts:{ea:'Microsoft YaHei'}},
 verifyArtifactToolImport:true,
 receiptPath:path.join(buildDir,'validation.json'),
});
console.log(JSON.stringify({finalPath:result.finalPath,slides:result.packageIntegrity.slide_count,layoutFindings:result.presentationLayout.finding_count,importPassed:result.firstPartyImport.passed}));
