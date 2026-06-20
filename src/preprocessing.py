#preprocessing
import re
import html

# --- CÁC REGEX LÀM SẠCH ---
DATELINE_PATTERNS = {
    'Reuters': r'^.{0,80}\(Reuters\)\s*[-–—:]+',
    'AP':      r'^.{0,80}\(AP\)\s*[-–—:]+',
    'AFP':     r'^.{0,80}\(AFP\)\s*[-–—:]+',
    'Generic': r'^[A-Z][A-Za-z\s,\.\d]{1,60}\((?:[A-Za-z]{2,15})\)\s*[-–—:]+',
}
REUTERS_CREDIT_LINE = r'\(Reporting by .{0,80}?\)\s*$'

COMPILED_DATELINES = [re.compile(pat, re.IGNORECASE) for pat in DATELINE_PATTERNS.values()]
COMPILED_CREDIT = re.compile(REUTERS_CREDIT_LINE, re.IGNORECASE)


def clean_html_and_urls(text):
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " URLTOKEN ", text)
    return re.sub(r"\s+", " ", text).strip()


def deep_unescape(text, max_iter=5):
    prev = text
    for _ in range(max_iter):
        new = html.unescape(prev)
        if new == prev:
            break
        prev = new
    return prev


def strip_dateline_and_credit(text, max_passes=5):
    cleaned = text
    for _ in range(max_passes):
        changed = False
        for pat in COMPILED_DATELINES:
            new_cleaned, n = pat.subn(' ', cleaned, count=1)
            if n > 0:
                cleaned, changed = new_cleaned, True
        if not changed:
            break
    cleaned = COMPILED_CREDIT.sub(' ', cleaned)
    return re.sub(r'\s+', ' ', cleaned).strip()


def preprocess_text(title, body):
    """
    Hàm chính để gọi khi dự đoán. 
    Nhận vào title và body gốc, trả về text đã được làm sạch.
    """
    title = str(title).strip() if title else ""
    body = str(body).strip() if body else ""
    
    body_clean = clean_html_and_urls(body)
    body_clean = deep_unescape(body_clean)
    body_final = strip_dateline_and_credit(body_clean)
    
    if body_final.strip() == '':
        body_final = title
        
    return title, body_final