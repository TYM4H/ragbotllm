import re

def clean_text(t: str) -> str:
    """Очищает текст от мусора и лишних символов."""
    t = re.sub(r'\s+', ' ', t)
    t = re.sub(r'[^\w\s.,;:!?()\-\[\]\(\)=+\/*°%]', '', t)
    return t.strip()
