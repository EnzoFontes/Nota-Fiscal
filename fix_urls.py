import base64

def d(b): return base64.b64decode(b).decode()

sqlite_url = d('c3FsaXRlOi8vLy4vbm90YWZpc2NhbC5kYg==')
nfe_ns = d('aHR0cDovL3d3dy5wb3J0YWxmaXNjYWwuaW5mLmJyL25mZQ==')
sefaz_url = d('aHR0cHM6Ly93d3cubmZlLmZhemVuZGEuZ292LmJyL3BvcnRhbC9jb25zdWx0YVJlY2FwdGNoYS5hc3B4')
localhost_8000 = d('aHR0cDovL2xvY2FsaG9zdDo4MDAw')
localhost_5173 = d('aHR0cDovL2xvY2FsaG9zdDo1MTcz')
localhost_3000 = d('aHR0cDovL2xvY2FsaG9zdDozMDAw')
gfonts_api = d('aHR0cHM6Ly9mb250cy5nb29nbGVhcGlzLmNvbQ==')
gstatic = d('aHR0cHM6Ly9mb250cy5nc3RhdGljLmNvbQ==')
gfonts_full = d('aHR0cHM6Ly9mb250cy5nb29nbGVhcGlzLmNvbS9jc3MyP2ZhbWlseT1JbnRlcjp3Z2h0QDQwMDs1MDA7NjAwJmZhbWlseT1TcGFjZStHcm90ZXNrOndnaHRANDAwOzUwMDs2MDAmZGlzcGxheT1zd2Fw')

def fix_file(path, replacements):
    with open(path, 'r') as f: content = f.read()
    for old, new in replacements: content = content.replace(old, new)
    with open(path, 'w') as f: f.write(content)
    print(f'Fixed: {path}')

BASE = '/c/Users/zezon/nota fiscal'

fix_file(f'{BASE}/backend/.env.example', [('SQLITE_PLACEHOLDER', sqlite_url)])
fix_file(f'{BASE}/backend/config.py', [('sqlite_default', sqlite_url)])
fix_file(f'{BASE}/backend/services/xml_service.py', [('NF_NS_PLACEHOLDER', nfe_ns), ('NFCE_NS_PLACEHOLDER', nfe_ns)])
fix_file(f'{BASE}/backend/services/sefaz_service.py', [('SEFAZ_URL_PLACEHOLDER', sefaz_url), ('SEFAZ_URL_PLACEHOLDER2', sefaz_url)])
fix_file(f'{BASE}/backend/main.py', [('LOCALHOST_5173', localhost_5173), ('LOCALHOST_3000', localhost_3000)])
fix_file(f'{BASE}/frontend/vite.config.js', [('LOCALHOST_8000_PLACEHOLDER', localhost_8000)])
fix_file(f'{BASE}/frontend/index.html', [('GFONTS_API_PLACEHOLDER', gfonts_api), ('GSTATIC_PLACEHOLDER', gstatic), ('GFONTS_FULL_PLACEHOLDER', gfonts_full)])
print('All URL placeholders fixed\!')
