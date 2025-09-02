

import os, json, csv, time, re
from collections import OrderedDict
from typing import Dict, Optional, Tuple

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR_SAME = BASE_DIR                          # data files beside bot.py
DATA_DIR_PARENT = os.path.join(BASE_DIR, "..", "data")  # or ../data

def existing_path(*candidates) -> str:
    for p in candidates:
        if p and os.path.exists(p):
            return p
    raise FileNotFoundError("None of the candidate paths exist:\n" +
                            "\n".join(c for c in candidates if c))

class LRUCache:
    def __init__(self, capacity=100):
        self.capacity = capacity
        self.store = OrderedDict()
    def get(self, key):
        if key in self.store:
            self.store.move_to_end(key)
            return self.store[key]
        return None
    def set(self, key, value):
        if key in self.store:
            self.store.move_to_end(key)
        self.store[key] = value
        if len(self.store) > self.capacity:
            self.store.popitem(last=False)

def sniff_delimiter(sample_line: str) -> str:
    if sample_line.count(";") > sample_line.count(","):
        return ";"
    return ","

def load_inventory(path: Optional[str] = None) -> Dict[str, Dict]:
    if path is None:
        c1 = os.path.join(DATA_DIR_SAME, "inventory.csv")
        c2 = os.path.join(DATA_DIR_PARENT, "inventory.csv")
        path = existing_path(c1, c2)

    inv: Dict[str, Dict] = {}
    with open(path, "rb") as fb:
        sample = fb.read(200).decode("utf-8", errors="ignore")
    delim = sniff_delimiter(sample)

    with open(path, newline="", encoding="utf-8-sig") as f:
        rdr_raw = csv.reader(f, delimiter=delim)
        rows = list(rdr_raw)

    if not rows:
        raise ValueError("inventory.csv is empty")

    headers = [h.strip().lower() for h in rows[0]]
    data_rows = []
    for r in rows[1:]:
        if len(r) < len(headers):
            r = r + [""] * (len(headers) - len(r))
        elif len(r) > len(headers):
            r = r[:len(headers)]
        data_rows.append({h: (v or "").strip() for h, v in zip(headers, r)})

    required = {"product_id", "name", "price", "stock_qty"}
    if not required.issubset(set(headers)):
        raise KeyError(f"inventory.csv missing required headers. "
                       f"Found: {headers} | Need: {sorted(required)}")

    for row in data_rows:
        name = row["name"].strip().lower()
        if not name:
            continue
        try:
            price = float(row.get("price", "0").replace(",", "."))
        except ValueError:
            price = 0.0
        try:
            stock = int(float(row.get("stock_qty", "0").replace(",", ".")))
        except ValueError:
            stock = 0
        inv[name] = {
            "product_id": row.get("product_id", "").strip(),
            "name": name,
            "price": price,
            "stock_qty": stock,
        }
    return inv

def load_faq(path: Optional[str] = None):
    if path is None:
        c1 = os.path.join(DATA_DIR_SAME, "faq.json")
        c2 = os.path.join(DATA_DIR_PARENT, "faq.json")
        path = existing_path(c1, c2)
    with open(path, encoding="utf-8-sig") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("faq.json must be a JSON array of {q,a} objects")
    out = []
    for item in data:
        if not isinstance(item, dict):
            continue
        q = str(item.get("q", "")).strip().lower()
        a = str(item.get("a", "")).strip()
        if q and a:
            out.append({"q": q, "a": a})
    return out

def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s

def keyword_overlap_score(q_tokens, cand_tokens):
    if not q_tokens or not cand_tokens:
        return 0.0
    inter = len(q_tokens.intersection(cand_tokens))
    denom = len(q_tokens) + 1e-6
    return inter / denom

def faq_fallback(user_text: str, faq_list, threshold=0.3) -> Optional[str]:
    q_tokens = set(re.findall(r"[a-z0-9]+", user_text.lower()))
    best_score, best_ans = 0.0, None
    for item in faq_list:
        cand_tokens = set(re.findall(r"[a-z0-9]+", item.get("q", "").lower()))
        score = keyword_overlap_score(q_tokens, cand_tokens)
        if score > best_score:
            best_score, best_ans = score, item.get("a", "")
    return best_ans if best_score >= threshold else None

def parse_numbers(user_text: str) -> Tuple[Optional[int], Optional[float]]:
    vals = [float(x.replace(",", ".")) for x in re.findall(r"\d+[\.,]?\d*", user_text)]
    qty = int(vals[0]) if len(vals) >= 1 else None
    price = float(vals[1]) if len(vals) >= 2 else None
    return qty, price

def calc_total(user_text: str) -> Optional[str]:
    qty, price = parse_numbers(user_text)
    if qty is not None and price is not None:
        total = qty * price
        return f"Your total is {total:.2f} for {qty} items at {price:.2f} each."
    return None

def extract_product(user_text: str) -> Optional[str]:
    m = re.search(r"(?:is|for|of|on)\s+([a-z0-9 ]+)", user_text.lower())
    if m:
        return m.group(1).strip()
    words = user_text.lower().split()
    if len(words) >= 2:
        return " ".join(words[-2:])
    return None

def check_stock(user_text: str, inventory: Dict[str, Dict]) -> Optional[str]:
    prod = extract_product(user_text)
    if not prod:
        return None
    candidates = []
    for name in inventory.keys():
        if prod in name or name in prod:
            candidates.append(name)
    if not candidates:
        p_tokens = set(prod.split())
        best, score = None, 0
        for name in inventory.keys():
            s = len(p_tokens.intersection(set(name.split())))
            if s > score:
                best, score = name, s
        if best:
            candidates = [best]
    if candidates:
        name = candidates[0]
        item = inventory[name]
        return (f"'{item['name']}' is in stock ({item['stock_qty']} units) at {item['price']:.2f}."
                if item["stock_qty"] > 0 else
                f"Sorry, '{item['name']}' is currently out of stock.")
    return "I could not find that product in inventory."

class AimlEngine:
    def __init__(self):
        self.rules = [
            ("hello", lambda t: "Hello! How can I help you today?"),
            ("hi", lambda t: "Hi! Ask me about products, prices, or stock."),
            ("hours", lambda t: "We’re open 24/7 online."),
            ("shipping", lambda t: "Standard shipping takes 3–5 business days."),
            ("return", lambda t: "Returns accepted within 30 days in original condition."),
            ("total", lambda t: calc_total(t) or "Say: total for 3 items at 25"),
            ("price", lambda t: calc_total(t) or "Tell me quantity and unit price to compute total."),
            ("stock", lambda t: "__STOCK__"),
            ("available", lambda t: "__STOCK__"),
        ]
    def respond(self, text: str):
        t = text.lower()
        for key, handler in self.rules:
            if key in t:
                return handler(text)
        return None

class ChatbotApp:
    def __init__(self,
                 inventory_path: Optional[str] = None,
                 faq_path: Optional[str] = None):
        self.inventory = load_inventory(inventory_path)
        self.faq = load_faq(faq_path)
        self.cache = LRUCache(capacity=100)
        self.aiml = AimlEngine()

    def handle_message(self, user_text: str) -> str:
        start = time.time()
        q_norm = normalize_text(user_text)

        cached = self.cache.get(q_norm)
        if cached:
            return f"(cached) {cached}"

        ans = self.aiml.respond(q_norm)
        if ans == "__STOCK__":
            ans = check_stock(q_norm, self.inventory)

        if ans and ans.strip():
            final = ans
        else:
            fb = faq_fallback(q_norm, self.faq, threshold=0.3)
            final = fb if fb else "Sorry, I didn’t catch that. Try shipping, returns, totals, or stock."

        self.cache.set(q_norm, final)
        ms = (time.time() - start) * 1000
        return f"{final} (response: {ms:.0f} ms)"

if __name__ == "__main__":
    print("Script directory (BASE_DIR):", BASE_DIR)
    try:
        inv_path = existing_path(
            os.path.join(DATA_DIR_SAME, "inventory.csv"),
            os.path.join(DATA_DIR_PARENT, "inventory.csv")
        )
        faq_path = existing_path(
            os.path.join(DATA_DIR_SAME, "faq.json"),
            os.path.join(DATA_DIR_PARENT, "faq.json")
        )
        print("Using inventory:", inv_path)
        print("Using FAQ:", faq_path)
    except FileNotFoundError as e:
        print("Data path error:", e)
        print("Place inventory.csv and faq.json either beside bot.py or in ../data/")
        raise

    app = ChatbotApp(inv_path, faq_path)
    print("Type your question (Ctrl+C to quit):")
    while True:
        try:
            msg = input("> ")
            print(app.handle_message(msg))
        except KeyboardInterrupt:
            print("\nBye!")
            break


