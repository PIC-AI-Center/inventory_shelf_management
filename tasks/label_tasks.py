from __future__ import annotations

from typing import Dict, List

from celery_app import celery_app


def _translate_batch(skus: List[str], source_lang: str, target_lang: str) -> Dict[str, str]:
    try:
        from google.cloud import translate_v2 as gct
        client = gct.Client()
        result: Dict[str, str] = {}
        for sku in skus:
            resp = client.translate(sku, source_language=source_lang, target_language=target_lang)
            result[sku] = resp["translatedText"]
        return result
    except ImportError:
        return {sku: sku for sku in skus}
    except Exception as exc:
        raise RuntimeError(f"Translation failed: {exc}") from exc


@celery_app.task(name="tasks.label_tasks.generate_label", bind=True)
def generate_label(
    self,
    skus: List[str],
    source_lang: str,
    target_lang: str,
) -> Dict[str, str]:
    self.update_state(state="STARTED", meta={"total": len(skus)})
    return _translate_batch(skus, source_lang, target_lang)
