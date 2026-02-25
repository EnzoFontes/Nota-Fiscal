import xml.etree.ElementTree as ET
from typing import Dict, List
from datetime import datetime

NS = 'http://www.portalfiscal.inf.br/nfe'


def _text(el, tag: str) -> str | None:
    child = el.find(f'{{{NS}}}{tag}') if el is not None else None
    return child.text if child is not None else None


def _float(val) -> float:
    try:
        return float(val) if val else 0.0
    except (ValueError, TypeError):
        return 0.0


def _parse_date(val: str | None) -> datetime | None:
    if not val:
        return None
    try:
        return datetime.fromisoformat(val[:19])
    except Exception:
        return None


def parse_nfe_xml(xml_bytes: bytes) -> Dict:
    root = ET.fromstring(xml_bytes)
    ide = root.find(f'.//{{{NS}}}ide')
    emit = root.find(f'.//{{{NS}}}emit')
    dest = root.find(f'.//{{{NS}}}dest')
    total = root.find(f'.//{{{NS}}}ICMSTot')
    info = root.find(f'.//{{{NS}}}infNFe')
    chave = info.get('Id', '').replace('NFe', '') if info is not None else None

    items: List[Dict] = []
    for det in root.findall(f'.//{{{NS}}}det'):
        prod = det.find(f'{{{NS}}}prod')
        if prod is not None:
            items.append({
                'descricao': _text(prod, 'xProd'),
                'ncm': _text(prod, 'NCM'),
                'quantidade': _float(_text(prod, 'qCom')),
                'unidade': _text(prod, 'uCom'),
                'valor_unitario': _float(_text(prod, 'vUnCom')),
                'valor_total': _float(_text(prod, 'vProd')),
                'cfop': _text(prod, 'CFOP'),
            })

    tributos: List[Dict] = []
    if total is not None:
        for tag, tipo in [('vICMS', 'ICMS'), ('vPIS', 'PIS'), ('vCOFINS', 'COFINS')]:
            val = _text(total, tag)
            if val:
                tributos.append({'tipo': tipo, 'valor': _float(val),
                                  'base_calculo': 0.0, 'aliquota': 0.0})

    dest_doc = (_text(dest, 'CNPJ') or _text(dest, 'CPF')) if dest is not None else None

    return {
        'tipo': 'nfe',
        'chave_acesso': chave,
        'numero': _text(ide, 'nNF'),
        'serie': _text(ide, 'serie'),
        'data_emissao': _parse_date(_text(ide, 'dhEmi')),
        'emitente_cnpj': _text(emit, 'CNPJ'),
        'emitente_razao_social': _text(emit, 'xNome'),
        'destinatario_cpf_cnpj': dest_doc,
        'destinatario_nome': _text(dest, 'xNome'),
        'valor_total': _float(_text(total, 'vNF')),
        'valor_desconto': _float(_text(total, 'vDesc')),
        'valor_frete': _float(_text(total, 'vFrete')),
        'itens': items,
        'tributos': tributos,
        'confidence_score': 100.0,
    }
