"""Prepare an exact public allowlist and build a new selected-method release.

No downloads, fitting or evaluation. --refresh-allowlist prepares the reviewable
inventory; --build is explicit and refuses to overwrite any prior release.
Provided data and rejected model assets are omitted. Their source and hashes
remain available through fetch_provided_data.py and PACKAGE_MANIFEST.json.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import zipfile
from freeze_selected import verify

ROOT = Path(__file__).resolve().parent.parent
ALLOWLIST = ROOT/'OPTIMIZED_RELEASE_ALLOWLIST.txt'
MAX_METADATA = 4*2**20
STUDIES = ('baselines','trees','process','sequence','regularization','interactions',
           'alternatives','sequence_aux','calibration','subset','bagging',
           'reference_augmentation','graph','stopping','transformer','fusion','finalists',
           'final_selection','reproduction_template')
METADATA = {'protocol.json','summary.json','verification.json','source_evidence.json',
            'fit_records.json','oof_schema.json','selected_recipe.json','shortlist.json',
            'resource_assessments.json','prevalidation_freeze.json','selection.json',
            'validation_results.json','contract_checks.json','contracts.json'}
FORBIDDEN = {'.git','.local','.venv','venv','__pycache__','materials','worklogs',
             'learning','logs','refits','folds','cache','features','fitting_scores'}
DATA_NOTICES = ('LICENSE.txt','NOTICE.txt','manifest.json')
REPORT_FILES = ('report.pdf','report_source.txt','evidence_manifest.json','claims.json',
                'report_build_audit.json','layout_checks.json')
PUBLIC_DOCS=('core_refit_protocol.md','optimization_refit_protocol.md','optimization_results.md',
             'optimization_protocol.md','optimization_selection_policy.md','provenance.md',
             'release.md','submission.md','method_notes.md','extended_protocol.md')
REPRODUCTION_TOOLS=('train_optimized.py','fetch_provided_data.py','check_data.py',
    'build_optimization_report.py','render_report_documents.py','prepare_optimized_report.py','analyze_optimized.py',
    'build_optimized_readme.py','package_optimized_release.py','verify_optimized_checkpoint.py',
    'optimization_validate_finalists.py')


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe_name(name):
    path = PurePosixPath(name)
    if not name or '\\' in name or ':' in name or path.is_absolute() or str(path)!=name:
        raise ValueError('Noncanonical public path: '+name)
    if any(part in ('','.','..') or part.rstrip(' .')!=part for part in path.parts):
        raise ValueError('Unsafe public path: '+name)
    if any(part.casefold() in FORBIDDEN for part in path.parts):
        raise ValueError('Private/transient path: '+name)
    return path


def inside(path):
    path = Path(path).resolve()
    if ROOT not in path.parents: raise ValueError('Release input must be inside the repository')
    return path.relative_to(ROOT).as_posix()


def local_source_closure(roots):
    """Only requested entry points and their static local Python imports."""
    found=set();todo=list(roots)
    while todo:
        name=todo.pop()
        if name in found:continue
        path=ROOT/'code'/name
        if not path.is_file():raise FileNotFoundError(path)
        found.add(name)
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'),filename=name)):
            modules=[]
            if isinstance(node,ast.Import):modules=[x.name.split('.')[0] for x in node.names]
            elif isinstance(node,ast.ImportFrom) and node.module:modules=[node.module.split('.')[0]]
            for module in modules:
                local=module+'.py'
                if (ROOT/'code'/local).is_file() and local not in found:todo.append(local)
    return sorted(found)


def collect(report_dir):
    frozen = verify(ROOT/'code/runs/optimized_freeze.json')
    names = set()
    def add(name, bounded=False):
        safe_name(name); path=ROOT/name
        if not path.is_file(): raise FileNotFoundError(path)
        if path.is_symlink() or ROOT not in path.resolve().parents: raise ValueError('Linked/escaping public file: '+name)
        if bounded and path.stat().st_size>MAX_METADATA: raise ValueError('Oversized metadata: '+name)
        names.add(name)
    for name in ('README.md','guide/GUIDE.md','code/README.md','code/PACKAGE_MANIFEST.json',
                 'code/configs/selected_recipe.json','code/configs/upstream_data.json',
                 'code/runs/optimization_final_analysis/report_inputs.json'):
        add(name)
    roots=set(frozen['source_sha256'])|set(REPRODUCTION_TOOLS)
    roots.update('optimization_'+stage+'.py' for stage in STUDIES
                 if (ROOT/'code'/('optimization_'+stage+'.py')).is_file())
    for name in local_source_closure(roots):add('code/'+name)
    for name in ('requirements.txt','requirements-train.txt','requirements-sequence.txt','requirements-report.txt'):
        add('code/'+name)
    for name in PUBLIC_DOCS:add('docs/'+name,True)
    for name in ('optimization_report_reference.json','report_content_en.json'):
        if (ROOT/'docs'/name).is_file(): add('docs/'+name,True)
    for name in DATA_NOTICES: add('code/data/'+name,True)
    for name in frozen['source_sha256']: add('code/'+name)
    for name in frozen['asset_inventory']['files']: add('code/'+frozen['checkpoint']+'/'+name)
    add('code/runs/optimized_freeze.json',True)
    add('code/'+frozen['selection_record'],True)
    for stage in STUDIES:
        folder=ROOT/'code/runs'/('optimization_'+stage)
        for name in sorted(METADATA):
            if (folder/name).is_file(): add(inside(folder/name),True)
    for name in ('evaluation_started.json','test_score.json','test_resources.json'):
        add('code/runs/optimization_evaluation/'+name,True)
    for name in ('summary.json','verification.json'):
        path=ROOT/'code/runs/optimized_reproduction_work'/name
        if path.is_file(): add(inside(path),True)
    # Exact small evidence dependencies are public only when the report itself
    # binds their hashes. Never pull in an unbounded working directory.
    relative_report=inside(report_dir)
    for name in REPORT_FILES: add(relative_report+'/'+name, name!='report.pdf')
    if (report_dir/'content_review.json').is_file():add(relative_report+'/content_review.json',True)
    for name in ('inputs.json','report_inputs.json'):
        if (report_dir/name).is_file(): add(relative_report+'/'+name,True)
    evidence=read(report_dir/'evidence_manifest.json')
    for record in evidence['sources']:
        name=record['path'];safe_name(name)
        path=ROOT/name
        if path.suffix.lower() not in ('.json','.md','.txt'):
            if name not in names: raise ValueError('Unreviewed report dependency type: '+name)
        else: add(name,True)
        if sha(path)!=record['sha256']: raise ValueError('Report evidence changed: '+name)
    # Epoch provenance required by the selected scratch CNN's fresh refit.
    for name in ('protocol.json','summary.json','fit_records.json','source_evidence.json'):
        add('code/runs/optimization_sequence/'+name,True)
    add('docs/optimization_refit_protocol.md',True)
    names.update((ALLOWLIST.name,'.gitignore'))
    if len({name.casefold() for name in names})!=len(names): raise ValueError('Case-colliding release paths')
    return sorted(names), frozen


def validate_report(report_dir):
    pdf=report_dir/'report.pdf'; build=read(report_dir/'report_build_audit.json'); layout=read(report_dir/'layout_checks.json')
    digest=sha(pdf)
    if pdf.read_bytes()[:5]!=b'%PDF-': raise ValueError('Invalid report PDF')
    for record in (build,layout):
        if record.get('pdf_sha256')!=digest or not 1<=record.get('pages',0)<=10:
            raise ValueError('Report page/hash evidence mismatch')
    if not layout.get('page_bounds_passed'): raise ValueError('Report bounds check missing/failed')
    if build['pages']!=layout['pages']: raise ValueError('Inconsistent report page counts')


def validate_upstream(names):
    originals=read(ROOT/'code/PACKAGE_MANIFEST.json')['files']
    replacements={'code/submission.py','code/requirements.txt'}
    for name in set(names)&set(originals)-replacements:
        if sha(ROOT/name)!=originals[name]: raise ValueError('Supplied immutable file changed: '+name)


def write_allowlist(names):
    ALLOWLIST.write_text('# Exact public files for the selected-method release.\n'+'\n'.join(names)+'\n',encoding='utf-8')
    previous=ROOT/'.gitignore.before-optimized'
    current=ROOT/'.gitignore'
    if current.is_file() and not previous.exists():
        with previous.open('xb') as stream:stream.write(current.read_bytes())
    directories={p.as_posix() for name in names for p in PurePosixPath(name).parents if p.as_posix()!='.'}
    lines=['# Exact public allowlist; caches and new files are private by default.','*']
    lines+=['!/'+name+'/' for name in sorted(directories,key=lambda n:(n.count('/'),n))]
    lines+=['!/'+name for name in names]
    current.write_text('\n'.join(lines)+'\n',encoding='utf-8')


def audit_archive(path,manifest):
    if sha(path)!=manifest['archive_sha256']: raise ValueError('Archive checksum mismatch')
    with zipfile.ZipFile(path) as archive:
        infos=archive.infolist();names=[item.filename for item in infos]
        if len(names)!=len(set(n.casefold() for n in names)) or set(names)!=set(manifest['files']):
            raise ValueError('Archive membership differs')
        for item in infos:
            safe_name(item.filename)
            if item.is_dir() or stat.S_ISLNK(item.external_attr>>16) or item.flag_bits&1:
                raise ValueError('Unexpected archive member type')
            value=manifest['files'][item.filename]
            if item.file_size!=value['bytes'] or hashlib.sha256(archive.read(item)).hexdigest()!=value['sha256']:
                raise ValueError('Archive member differs: '+item.filename)
    return {'passed':True,'files':len(manifest['files']),'archive_sha256':manifest['archive_sha256']}


def archive(path,files,kind):
    if path.exists() or path.with_suffix('.manifest.json').exists() or path.with_suffix('.audit.json').exists():
        raise FileExistsError('Preserve prior release: '+str(path))
    records={name:{'sha256':sha(source),'bytes':source.stat().st_size} for name,source in files.items()}
    with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as stream:
        for name,source in sorted(files.items()): stream.write(source,safe_name(name).as_posix())
    manifest={'schema':2,'kind':kind,'archive':path.name,'archive_sha256':sha(path),
              'archive_bytes':path.stat().st_size,'files':records,'file_count':len(records)}
    with path.with_suffix('.manifest.json').open('x',encoding='utf-8') as stream: json.dump(manifest,stream,indent=2)
    result=audit_archive(path,manifest)
    with path.with_suffix('.audit.json').open('x',encoding='utf-8') as stream: json.dump(result,stream,indent=2)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir',type=Path,default=ROOT/'report')
    parser.add_argument('--output-dir',type=Path,default=ROOT.parent/'materials/submission')
    parser.add_argument('--refresh-allowlist',action='store_true')
    parser.add_argument('--build',action='store_true')
    parser.add_argument('--separate-checkpoint',action='store_true')
    parser.add_argument('--audit',type=Path,help='Audit an existing ZIP against its sidecar; no other work')
    args=parser.parse_args()
    if args.audit:
        print(json.dumps(audit_archive(args.audit,read(args.audit.with_suffix('.manifest.json'))),indent=2));return
    report=args.report_dir.resolve()
    names,frozen=collect(report);validate_report(report);validate_upstream(names)
    if args.refresh_allowlist:
        write_allowlist(names)
    recorded=[s for s in ALLOWLIST.read_text(encoding='utf-8').splitlines() if s and not s.startswith('#')] if ALLOWLIST.exists() else []
    if recorded!=names: raise RuntimeError('Review files, then run --refresh-allowlist before building')
    plan={'status':'prepared; no archives created','files':names,'file_count':len(names),
          'file_bytes':sum((ROOT/name).stat().st_size for name in names),
          'by_top_directory':dict(Counter(PurePosixPath(n).parts[0] for n in names)),
          'checkpoint_uncompressed_bytes':frozen['asset_inventory']['uncompressed_bytes'],
          'provided_data_included':False,'private_worklogs_included':False,
          'freeze_sha256':sha(ROOT/'code/runs/optimized_freeze.json')}
    print(json.dumps(plan,indent=2))
    if not args.build:return
    output=args.output_dir.resolve();output.mkdir(parents=True,exist_ok=True)
    files={name:ROOT/name for name in names}
    release=archive(output/'submission.zip',files,'selected_project')
    if args.separate_checkpoint:
        directory=ROOT/'code'/frozen['checkpoint']
        archive(output/'checkpoint.zip',{name:directory/name for name in frozen['asset_inventory']['files']},'selected_checkpoint')
    print(json.dumps({'status':'built and hash-audited','archive':release['archive'],'bytes':release['archive_bytes']}))


if __name__=='__main__':main()
