import httpx
import logging
from config import SEFAZ_TIMEOUT

logger = logging.getLogger(__name__)


async def validate_chave_acesso(chave: str) -> dict:
    """Query SEFAZ to validate a NF-e access key (44 digits)."""
    if not chave or len(chave) != 44:
        return {"status": "nao_aplicavel", "message": "Chave inválida ou ausente"}
    try:
        async with httpx.AsyncClient(timeout=SEFAZ_TIMEOUT) as client:
            resp = await client.get(
                "https://www.nfe.fazenda.gov.br/portal/consultaRecaptcha.aspx",
                params={"tipoConteudo": ".faces", "chNFe": chave},
                follow_redirects=True,
            )
            if resp.status_code == 200 and "Autorizado" in resp.text:
                return {"status": "validado", "message": "Autorizado o uso da NF-e"}
            return {"status": "nao_validado", "message": "Não validado pela SEFAZ"}
    except httpx.TimeoutException:
        logger.warning(f"SEFAZ timeout — chave {chave[:10]}...")
        return {"status": "falha",
                "message": "Timeout ao consultar SEFAZ — processado sem validação"}
    except Exception as e:
        logger.error(f"SEFAZ error: {e}")
        return {"status": "falha", "message": str(e)}
