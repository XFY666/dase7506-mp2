"""Render a structured, evidence-bound report in English or Chinese.

This document tool neither trains a model nor evaluates predictions.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from xml.sax.saxutils import escape

from build_optimization_report import load_inputs


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--content', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--font-body', type=Path)
    parser.add_argument('--font-bold', type=Path)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    spec, evidence, final, *_ = load_inputs(args.inputs)
    document = json.loads(args.content.read_text(encoding='utf-8'))
    language = document['language']
    if language not in ('en', 'zh') or len(document['sections']) != 9:
        raise ValueError('Expected an English or Chinese report with nine sections')
    for section in document['sections']:
        if not section.get('title') or not section.get('blocks'):
            raise ValueError('Empty report section')
        for block in section['blocks']:
            if block['type'] not in ('paragraph', 'equation', 'table'):
                raise ValueError('Unsupported report block')
            if not block.get('evidence') or any(k not in evidence.records for k in block['evidence']):
                raise ValueError('Every block must identify verified evidence')
            if block['type'] == 'table':
                columns = len(block['headers'])
                if any(len(row) != columns for row in block['rows']):
                    raise ValueError('Unequal table row length')
                widths = block.get('widths', [1/columns]*columns)
                if len(widths) != columns or abs(sum(widths)-1)>1e-6 or min(widths)<=0:
                    raise ValueError('Table widths must be positive fractions summing to one')
            elif not block.get('text'):
                raise ValueError('Empty report text')
    if args.validate_only:
        print(json.dumps(dict(status='content_and_evidence_valid', language=language, sections=9)))
        return
    out = args.output.resolve()
    if (out/'report.pdf').exists():
        raise FileExistsError('Preserve the existing report before rendering a replacement')
    out.mkdir(parents=True, exist_ok=True)
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer, PageBreak
    if language == 'zh':
        if args.font_body is None or args.font_bold is None:
            raise ValueError('Chinese rendering requires explicit local font paths')
        pdfmetrics.registerFont(TTFont('ReportCJK', str(args.font_body), subfontIndex=0))
        pdfmetrics.registerFont(TTFont('ReportCJKBold', str(args.font_bold), subfontIndex=0))
        font, bold = 'ReportCJK', 'ReportCJKBold'
    else:
        font, bold = 'Helvetica', 'Helvetica-Bold'
    styles = {
        'title': ParagraphStyle('ReportTitle', fontName=bold, fontSize=22, leading=29, spaceAfter=10, textColor=colors.HexColor('#123d59')),
        'subtitle': ParagraphStyle('ReportSubtitle', fontName=font, fontSize=11, leading=16, spaceAfter=12, textColor=colors.HexColor('#4d6175')),
        'heading': ParagraphStyle('ReportHeading', fontName=bold, fontSize=17, leading=24, spaceAfter=12, textColor=colors.HexColor('#123d59')),
        'body': ParagraphStyle('ReportBody', fontName=font, fontSize=10, leading=15 if language=='zh' else 13.6, spaceAfter=8, wordWrap='CJK' if language=='zh' else None),
        'table': ParagraphStyle('ReportTable', fontName=font, fontSize=8.2, leading=12 if language=='zh' else 10.6, wordWrap='CJK' if language=='zh' else None),
        'table_header': ParagraphStyle('ReportTableHeader', fontName=bold, fontSize=8.2, leading=12 if language=='zh' else 10.6, wordWrap='CJK' if language=='zh' else None),
        'caption': ParagraphStyle('ReportCaption', fontName=font, fontSize=8.3, leading=12, spaceAfter=8, textColor=colors.HexColor('#4d6175'), wordWrap='CJK' if language=='zh' else None),
        'equation': ParagraphStyle('ReportEquation', fontName=font, fontSize=10, leading=15, leftIndent=12, spaceAfter=10, textColor=colors.HexColor('#123d59'))
    }
    def clean(value):
        text=str(value)
        if language=='en':
            text=text.replace('\u2011','-').replace('\u2013','-').replace('\u2014','-')
        return escape(text).replace('\n','<br/>')
    flow, transcript, claims = [], [], []
    flow.append(Paragraph(clean(document['title']), styles['title']))
    flow.append(Paragraph(clean(document.get('subtitle','DASE7506 Mini-project 2')), styles['subtitle']))
    transcript.extend((document['title'], document.get('subtitle','')))
    for si, section in enumerate(document['sections']):
        if si: flow.append(PageBreak())
        flow.append(Paragraph(clean(section['title']), styles['heading']))
        transcript.append('\n'+section['title'])
        for bi, block in enumerate(section['blocks']):
            claims.append(dict(section=si+1, block=bi+1, kind=block['type'],
                text_sha256=hashlib.sha256(json.dumps(block,sort_keys=True,ensure_ascii=False).encode()).hexdigest(),
                evidence=block['evidence'],human_verified=False))
            if block['type']=='table':
                n=len(block['headers']); widths=block.get('widths',[1/n]*n)
                data=[[Paragraph(clean(x),styles['table_header']) for x in block['headers']]]
                data += [[Paragraph(clean(x),styles['table']) for x in row] for row in block['rows']]
                table=Table(data,colWidths=[516*x for x in widths],repeatRows=1,hAlign='LEFT')
                table.setStyle(TableStyle([
                    ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8f0f5')),
                    ('VALIGN',(0,0),(-1,-1),'TOP'),
                    ('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#527287')),
                    ('LINEBELOW',(0,-1),(-1,-1),.4,colors.HexColor('#a7b5c0')),
                    ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),
                ]))
                flow.extend((table,Spacer(1,8)))
                transcript.append(' | '.join(map(str,block['headers'])))
                transcript.extend(' | '.join(map(str,row)) for row in block['rows'])
                if block.get('caption'):
                    flow.append(Paragraph(clean(block['caption']),styles['caption']))
                    transcript.append(block['caption'])
            else:
                flow.append(Paragraph(clean(block['text']),styles['equation' if block['type']=='equation' else 'body']))
                transcript.append(block['text'])
    page_numbers=[]
    def footer(canvas, doc):
        page_numbers.append(doc.page)
        canvas.saveState();canvas.setFont(font,8);canvas.setFillColor(colors.HexColor('#607080'))
        canvas.drawString(48,28,'DASE7506 | Mini-project 2')
        canvas.drawRightString(564,28,str(doc.page));canvas.restoreState()
    path=out/'report.unverified.pdf'
    pdf=SimpleDocTemplate(str(path),pagesize=(612,792),leftMargin=48,rightMargin=48,topMargin=42,bottomMargin=45,
        title=document['title'],author='')
    pdf.build(flow,onFirstPage=footer,onLaterPages=footer)
    pages=max(page_numbers)
    if pages>10:
        raise RuntimeError(f'Report has {pages} pages; repair layout before delivery')
    path.replace(out/'report.pdf')
    (out/'report_source.txt').write_text('\n\n'.join(transcript)+'\n',encoding='utf-8')
    evidence.add('E_REPORT_CONTENT_'+language.upper(),args.content,'Report text, structured tables and evidence references')
    manifest=dict(sources=list(evidence.records.values()),human_review='pending',submission_ready=False)
    audit=dict(status='programmatic_checks_passed',language=language,pages=pages,maximum_pages=10,
        pdf_sha256=sha(out/'report.pdf'),builder_sha256=sha(__file__),
        evidence_validator_sha256=sha(Path(__file__).with_name('build_optimization_report.py')),
        inputs_sha256=sha(args.inputs),content_sha256=sha(args.content),
        visual_review='pending',human_scientific_review='pending',submission_ready=False,
        created_utc=datetime.now(timezone.utc).isoformat())
    for name,value in [('evidence_manifest.json',manifest),('claims.json',claims),('report_build_audit.json',audit)]:
        (out/name).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(audit))


if __name__=='__main__':
    main()
