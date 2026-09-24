"""Sinh dữ liệu BIO L1–L7 theo phong cách golden để fine-tune kiendt/phobert-ner-address.

Định dạng mỗi dòng giống `output_model.txt` (text, tokens, bio, trunc, spans, len_nfc, is_full,
source_id, cut, augment, group, n_anchor, is_long). Ghi ra models/data/phobert_ner/:
train.jsonl (80%), test.jsonl (20%), meta.json (cấu hình + thống kê so với golden).

Nguồn: address_db (18k bản ghi hành chính + tên đường OSM) của repo anh em address-parser-short,
dùng lại tokenizer và các phép nhiễu của nó. Thêm ở đây những gì golden có mà address_db thiếu:
POI đa dạng (dự án, toà nhà, trường, bệnh viện, quán...), số nhà ngõ/hẻm/khoảng, nhiều cách viết
tỉnh, bỏ dấu phẩy, viết thường toàn câu.

Mỗi mẫu:
  1. dựng địa chỉ đầy đủ từ một bản ghi — nhãn biết trước theo cách dựng, không đoán lại
  2. nhiễu hoá trên danh sách token (viết tắt, mất dấu, typo, hoa/thường) — nhãn không lệch
  3. 20% giữ nguyên câu đầy đủ, 80% cắt thành tiền tố gõ dở (tỉ lệ 150:600 của golden_full:golden_uncomplete)

`truncated` theo prompt §3.5 — chỉ true khi CHÍNH VĂN BẢN có bằng chứng, không phải vì generator
biết chuỗi gốc dài hơn:
  * cắt giữa một token chữ ("Quận Li", "Xuâ", "Hải P")
  * số nhà có "/" hoặc "-" treo ở cuối ("125/", "17 -")
  * span chỉ còn từ chỉ thị ("858 Đường", "Q.", "Đại học", "Quán cà phê")
  "239" cắt từ "2395" hay "Đường Nguyễn" cắt từ "Đường Nguyễn Huệ" -> false.

Chống rò rỉ:
  * bản ghi address_db là nguồn của một câu golden bị loại khỏi cả train lẫn test
  * POI trùng POI golden bị sinh lại; mẫu trùng text golden bị bỏ
  * chia train/test theo source_id; bản ghi street_pool=heldout chỉ vào test (tên đường chưa thấy khi train)
  * mẫu test trùng text với train bị bỏ

    python scripts/build_phobert_ner_dataset.py                                   # 100k, seed 42
    python scripts/build_phobert_ner_dataset.py --n 2000 --out models/data/phobert_ner_small
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PARSER_REPO = ROOT.parent / "address-parser-short"
GOLDEN_DIR = ROOT / "golden_dataset"
GOLDEN_FILES = ["golden_full.json", "golden_uncomplete.json", "golden_test.json", "golden_test2.json"]

LEVELS = [f"L{i}" for i in range(1, 8)]

# ------------------------------------------------------------------ tỉ lệ (hiệu chỉnh theo golden_full)
FULL_RATIO = 0.20              # golden_full : golden_uncomplete = 150 : 600
POI_RATIO = 0.42               # 68/150 câu golden_full có L7
POI_AFTER_STREET = 0.12        # 6/61 câu có POI đứng sau số nhà/đường
COUNTRY_RATIO = 0.05
# Tổ hợp level (chưa tính L7/L1), trọng số theo bảng tổ hợp của golden_full
STRUCTURES = [
    (("L6", "L5", "L4", "L3", "L2"), 0.34), (("L6", "L5", "L3", "L2"), 0.16),
    (("L5", "L4", "L3", "L2"), 0.10), (("L6", "L5", "L4", "L3"), 0.10),
    (("L6", "L5", "L4", "L2"), 0.08), (("L6", "L5", "L2"), 0.05), (("L6", "L5"), 0.04),
    (("L5", "L4", "L3"), 0.03), (("L5", "L3", "L2"), 0.03), (("L3", "L2"), 0.03),
    (("L6", "L4", "L2"), 0.02), (("L4", "L3", "L2"), 0.02),
]
SEP_MODES = [("comma", 0.62), ("none", 0.30), ("mixed", 0.08)]    # golden: 46/150 câu không dấu phẩy
CASING = [("keep", 0.29), ("lower", 0.63), ("upper", 0.02), ("title", 0.06)]  # golden: 93/150 viết thường
AUG = {"abbreviate": 0.10, "strip_diacritics": 0.04, "typo": 0.05}
# Tiền tố gõ dở: loại điểm cắt và độ dài (golden_uncomplete: trung vị 16 ký tự)
CUT_KINDS = [("span_end", 0.30), ("mid_token", 0.47), ("token_end", 0.23)]
CUT_DECAY = 16.0
MIN_PREFIX = 2

# Bản ghi OSM có tên "đường" là nhiễu (lối đi, nhánh, kho...) — bỏ
NOISY_STREET_HEADS = {"Đi", "Hèm", "Nhánh", "Lối", "Kho", "Dãy"}

# ------------------------------------------------------------------ từ vựng POI (không lấy từ golden)
EN_WORDS = [
    "Sun", "Sunrise", "Sunset", "Star", "River", "Riverside", "Green", "Golden", "Silver", "Diamond",
    "Royal", "Park", "Sky", "Ocean", "Lake", "Pearl", "Sapphire", "Emerald", "Lotus", "Orchid",
    "Capital", "Central", "Grand", "Horizon", "Times", "Heights", "Palace", "Eco", "Smart", "Happy",
    "Harmony", "Luxury", "Premier", "Elite", "Crystal", "Moonlight", "Saigon", "Hanoi", "Phoenix",
    "Maple", "Victoria", "Imperial", "Galaxy", "Aqua", "Vista", "Season", "Lucky", "Charm", "Rose",
    "Jade", "Ruby", "Topaz", "Opera", "Metro", "Urban", "Garden", "Hill", "Bay", "Coral", "Sakura",
]
EN_SUFFIX = [
    "Tower", "Plaza", "Residence", "City", "Building", "Center", "Complex", "Villas", "Home", "Garden",
    "Park", "Apartment", "Heights", "Square", "Landmark", "Court", "Mall", "Office", "Suites", "Land",
]
VN_BRANDS = [
    "Hưng Thịnh", "Phát Đạt", "Thành Công", "Hoàng Anh", "Minh Long", "Phú Mỹ", "An Phú", "Hoà Bình",
    "Sao Mai", "Ánh Dương", "Bình Minh", "Hồng Hà", "Kim Long", "Thiên Phú", "Phúc Thịnh", "Vạn Phúc",
    "Đông Á", "Việt Hưng", "Hải Âu", "Thăng Long", "Sông Đà", "Hà Đô", "Văn Phú", "Gia Phát",
    "Hưng Phát", "Lộc Phát", "Thanh Bình", "Hoàng Quân", "Đất Xanh", "Tân Hoàng", "Phú Gia", "Mỹ Đình",
]
BUILDING_CODES = ["CT1", "CT2A", "CT5", "HH3", "HH4C", "A1", "B2", "C5", "N04", "R2", "T1", "S3",
                  "17T5", "24T2", "Block B", "Tháp A", "Tháp B", "Khu A"]
UNI_FIELDS = [
    "Bách khoa", "Kinh tế", "Ngoại thương", "Sư phạm", "Y Dược", "Công nghiệp", "Giao thông Vận tải",
    "Kiến trúc", "Luật", "Mỹ thuật", "Nông Lâm", "Thủy lợi", "Xây dựng", "Văn hóa", "Tài chính - Marketing",
    "Ngân hàng", "Khoa học Tự nhiên", "Mở", "Quốc tế", "Công nghệ", "Tôn Đức Thắng", "Văn Lang",
]
HOSPITAL_NAMES = ["Nhi Đồng 1", "Nhi Đồng 2", "Phụ sản", "Mắt", "Da liễu", "Ung bướu", "Tim", "Hoàn Mỹ",
                  "Tâm Anh", "Hạnh Phúc", "An Bình", "Thống Nhất", "Đại học Y", "Quân y 175", "Việt Pháp"]
SUPERMARKETS = ["Co.opmart", "Bách Hóa Xanh", "WinMart", "Big C", "Lotte Mart", "Mega Market", "Aeon",
                "Emart", "Satra", "GO!"]
BRANDS_BARE = ["Highlands Coffee", "Phúc Long", "The Coffee House", "Starbucks", "Cộng Cà Phê",
               "Trung Nguyên Legend", "Circle K", "FamilyMart", "GS25", "KFC", "Lotteria", "Jollibee",
               "Pizza Hut", "Điện Máy Xanh", "Thế Giới Di Động", "FPT Shop", "Pharmacity", "Long Châu"]
BANKS = ["Vietcombank", "Techcombank", "BIDV", "Agribank", "ACB", "VPBank", "MB", "Sacombank", "TPBank"]
KIN_NAMES = ["Cô Ba", "Chú Tư", "Bà Năm", "Dì Bảy", "Anh Tuấn", "Chị Hai", "O Lan", "Bác Sáu", "Cô Mười",
             "Ông Tám", "Mợ Hai", "Út Lan", "Thím Ba", "Bà Tư", "Cậu Út", "Chú Hải", "Cô Hạnh"]
POETIC = ["Mây Trắng", "Góc Phố", "Lá Me", "Hoa Sữa", "Gió Biển", "Phố Cũ", "Nắng Mai", "Sóng Xanh",
          "Bếp Nhà", "Hương Quê", "Ngọc Lan", "Chiều Tím", "Mộc Miên", "Xưa", "Làng Nướng", "Sen Hồng",
          "Trăng Non", "Bờ Sông", "Ruộng Lúa", "Hạt Dẻ", "Tre Xanh", "Mái Ngói", "Cây Bàng", "Ốc Đảo"]
FOOD_TYPES = ["Quán cà phê", "Cà phê", "Quán cơm tấm", "Cơm tấm", "Quán phở", "Phở", "Quán bún bò",
              "Bún chả", "Nhà hàng", "Tiệm trà sữa", "Quán ốc", "Quán lẩu", "Quán nhậu", "Tiệm bánh mì",
              "Bánh mì", "Quán trà đá", "Quán nước mía", "Quán bánh xèo", "Quán chè", "Tiệm cơm",
              "Nhà hàng hải sản", "Quán bún riêu", "Quán hủ tiếu", "Tiệm bánh"]
PROJECT_TYPES = ["dự án", "Dự án", "tòa nhà", "Tòa nhà", "toà nhà", "chung cư", "Chung cư",
                 "khu đô thị", "Khu đô thị", "KĐT", "khu dân cư", "Khu dân cư", "KDC", "căn hộ"]
SCHOOL_TYPES = ["Trường Đại học", "trường đại học", "Đại học", "Trường THPT", "Trường THCS",
                "Trường Tiểu học", "Trường Trung học phổ thông", "Trường Cao đẳng", "Học viện", "Trường Mầm non"]
HEALTH_TYPES = ["Bệnh viện", "bệnh viện", "Bệnh viện Đa khoa", "Phòng khám", "Phòng khám Đa khoa",
                "Nha khoa", "Trạm y tế", "Nhà thuốc"]
PUBLIC_TYPES = ["Chợ", "chợ", "Siêu thị", "Trung tâm thương mại", "Nhà thờ", "Chùa", "Đình", "Công viên",
                "Sân vận động", "Bến xe", "Ga", "Bưu điện", "Ngân hàng", "Khách sạn", "khách sạn", "Homestay",
                "Resort", "Nhà văn hóa", "Trung tâm", "Cửa hàng", "Showroom", "Công ty"]
POI_TYPE_PHRASES = PROJECT_TYPES + SCHOOL_TYPES + HEALTH_TYPES + PUBLIC_TYPES + FOOD_TYPES + ["tòa", "Tòa"]

# Từ chỉ thị theo level, dạng không dấu chữ thường — span chỉ còn (tiền tố của) cụm này ở cuối
# chuỗi là "tiền tố trơ trọi" -> truncated (prompt §3.5).
INDICATORS: dict[str, list[str]] = {
    "L1": [],
    "L2": ["thanh pho", "tp", "tinh", "t", "thu do"],
    "L3": ["quan", "q", "huyen", "h", "thi xa", "tx", "thanh pho", "tp"],
    "L4": ["phuong", "p", "xa", "x", "thi tran", "tt"],
    "L5": ["duong", "d", "pho", "dai lo", "quoc lo", "ql", "tinh lo", "huong lo", "so", "duong so"],
    "L6": ["so", "so nha", "ngo", "ngach", "hem", "kiet", "lo"],
    "L7": [],       # điền từ POI_TYPE_PHRASES khi chạy
}

WORD_RE = re.compile(r"[^\W\d_]", re.UNICODE)       # có ít nhất một chữ cái


# ------------------------------------------------------------------ tiện ích
def pick(rng: random.Random, table):
    return rng.choices([t for t, _ in table], weights=[w for _, w in table])[0]


def gnorm(s: str, nfc, strip_diacritics) -> str:
    """Khoá so khớp với golden: NFC, thường, không dấu, bỏ dấu câu (giữ '/')."""
    s = strip_diacritics(nfc(s).lower())
    s = re.sub(r"[^\w/]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


class Builder:
    def __init__(self, parser_repo: Path, seed: int) -> None:
        sys.path.insert(0, str(parser_repo))
        from model_address_parser.common import text as T
        from model_address_parser.common.gazetteer import Gazetteer, norm_key
        from model_address_parser.common.grouping import assign_group, n_anchor
        from model_address_parser.common.jsonlio import load_jsonl
        from model_address_parser.common.tagset import is_valid_bio
        from model_address_parser.pipeline.data import augment as A

        self.T, self.A = T, A
        self.norm_key = norm_key
        self.assign_group, self.n_anchor, self.is_valid_bio = assign_group, n_anchor, is_valid_bio
        pkg = parser_repo / "model_address_parser" / "pipeline" / "data"
        self.records = [r for r in load_jsonl(pkg / "synthetic" / "address_db.jsonl")
                        if r["street"].split()[0] not in NOISY_STREET_HEADS]
        self.gaz = Gazetteer.load(pkg / "gazetteer.v1.json")
        self.rng = random.Random(seed)

        self.key = lambda s: gnorm(s, T.nfc, T.strip_diacritics)
        INDICATORS["L7"] = sorted({self.key(p) for p in POI_TYPE_PHRASES})
        self._load_golden()
        self._build_pools()
        self._build_abbrev()

    # -------------------------------------------------------------- golden
    def _load_golden(self) -> None:
        texts, pois = [], set()
        for name in GOLDEN_FILES:
            p = GOLDEN_DIR / name
            if not p.exists():
                continue
            raw = json.loads(p.read_text(encoding="utf-8"))
            texts += raw["texts"]
            for r in raw["results"]:
                pois |= {self.key(x) for x in r["result"].get("L7", [])}
        self.golden_texts = {self.key(t) for t in texts}
        self.golden_pois = pois
        full = sorted(self.golden_texts)
        full_nospace = [t.replace(" ", "") for t in full]

        # Bản ghi nào là nguồn của một câu golden: số nhà + tên đường cùng xuất hiện liền nhau.
        excluded: set[int] = set()
        for r in self.records:
            house = re.sub(r"^(so|hem|ngo|ngach|kiet|lo) ", "", self.key(r["house"]))
            street = re.sub(r"^(duong|pho|d) ", "", self.key(r["street"]))
            if not house or not street:
                continue
            pat = re.compile(rf"(^| ){re.escape(house)} (duong |pho |d )?{re.escape(street)}( |$)")
            pat_ns = house.replace(" ", "") + street.replace(" ", "")
            for g, gn in zip(full, full_nospace):
                if (street.split()[0] in g and pat.search(g)) or (pat_ns in gn and g.startswith(house.split()[0])):
                    excluded.add(r["id"])
                    break
        self.excluded_ids = excluded
        self.records = [r for r in self.records if r["id"] not in excluded]

    # -------------------------------------------------------------- pools cho POI
    def _build_pools(self) -> None:
        person, places = set(), set()
        for r in self.records:
            s = re.sub(r"^(Đường|Phố)\s+", "", r["street"])
            parts = s.split()
            if 2 <= len(parts) <= 4 and all(p[:1].isupper() and p.isalpha() for p in parts):
                person.add(s)
            if not re.search(r"\d", r["ward"]):
                places.add(r["ward"])
        self.person_names = sorted(person)
        self.place_names = sorted(places)

    def _build_abbrev(self) -> None:
        """canonical -> alias, bỏ alias kiểu chữ cái đầu ("xcs", "dhs") ở L3–L7 — người dùng không gõ vậy."""
        rev: dict[str, list[str]] = {}
        for alias, target in self.gaz.alias.items():
            rev.setdefault(self.norm_key(target), []).append(alias)
        self.rev_alias = rev

    # -------------------------------------------------------------- bề mặt từng level
    def poi(self) -> str:
        rng = self.rng
        for _ in range(20):
            kind = pick(rng, [("project", 0.36), ("school", 0.12), ("health", 0.08), ("public", 0.14),
                              ("food", 0.20), ("brand", 0.10)])
            if kind == "project":
                t = rng.choice(PROJECT_TYPES)
                form = pick(rng, [("en2", 0.3), ("the", 0.15), ("vn", 0.25), ("code", 0.15), ("en1", 0.15)])
                if form == "en2":
                    name = f"{rng.choice(EN_WORDS)} {rng.choice(EN_SUFFIX)}"
                elif form == "the":
                    name = f"The {rng.choice(EN_WORDS)} {rng.choice(EN_WORDS)}"
                elif form == "vn":
                    name = f"{rng.choice(VN_BRANDS)} {rng.choice(EN_SUFFIX)}" if rng.random() < 0.5 \
                        else rng.choice(VN_BRANDS)
                elif form == "code":
                    name = rng.choice(BUILDING_CODES)
                    t = rng.choice(["tòa nhà", "Tòa nhà", "tòa", "chung cư", "Chung cư"])
                else:
                    name = f"{rng.choice(EN_WORDS)}{rng.choice(['', ' ' + str(rng.randint(1, 9))])}"
                if rng.random() < 0.15:
                    name += f" {rng.randint(1, 5)}"
                s = name if (form != "code" and rng.random() < 0.3) else f"{t} {name}"
            elif kind == "school":
                t = rng.choice(SCHOOL_TYPES)
                name = rng.choice(UNI_FIELDS) if t.lower().endswith(("đại học", "học viện", "cao đẳng")) \
                    and rng.random() < 0.6 else rng.choice(self.person_names)
                s = f"{t} {name}"
            elif kind == "health":
                t = rng.choice(HEALTH_TYPES)
                name = rng.choice(HOSPITAL_NAMES) if rng.random() < 0.5 else rng.choice(self.place_names)
                s = f"{t} {name}"
            elif kind == "public":
                t = rng.choice(PUBLIC_TYPES)
                if t == "Siêu thị":
                    name = rng.choice(SUPERMARKETS)
                elif t == "Ngân hàng":
                    name = rng.choice(BANKS)
                elif t in ("Khách sạn", "khách sạn", "Homestay", "Resort", "Showroom", "Công ty"):
                    name = rng.choice([rng.choice(VN_BRANDS), rng.choice(EN_WORDS), rng.choice(POETIC)])
                else:
                    name = rng.choice(self.place_names + self.person_names[:200])
                s = f"{t} {name}"
            elif kind == "food":
                name = rng.choice(KIN_NAMES + POETIC) if rng.random() < 0.8 else rng.choice(self.place_names)
                s = f"{rng.choice(FOOD_TYPES)} {name}"
            else:
                s = rng.choice(BRANDS_BARE + SUPERMARKETS)
            if self.key(s) not in self.golden_pois:
                return s
        return s

    def house(self, rec: dict) -> str:
        rng = self.rng
        n = lambda hi=999: str(rng.randint(1, hi))                       # noqa: E731
        form = pick(rng, [("orig", 0.40), ("num", 0.18), ("ngo", 0.12), ("hem", 0.07), ("so", 0.05),
                          ("lo", 0.03), ("range", 0.05), ("compound", 0.05), ("slash", 0.05)])
        if form == "orig":
            return rec["house"]
        if form == "num":
            return n() + (rng.choice("ABCDEabcde") if rng.random() < 0.15 else "")
        if form == "ngo":
            return rng.choice(["ngõ", "Ngõ", "ngách"]) + " " + (n(700) if rng.random() < 0.75 else f"{n(300)}/{n(80)}")
        if form == "hem":
            return rng.choice(["hẻm", "Hẻm", "kiệt"]) + " " + ("/".join(n(400) for _ in range(rng.choice([1, 1, 2, 3]))))
        if form == "so":
            return rng.choice(["số", "Số", "số nhà"]) + " " + n(400)
        if form == "lo":
            return rng.choice(["Lô", "lô"]) + " " + rng.choice("ABCDEFabcd") + n(40)
        if form == "range":
            a = rng.randint(1, 1500)
            k = rng.choice([2, 2, 2, 3])
            joiner = rng.choice([" - ", " - ", "-"])
            return joiner.join(str(a + 2 * i) for i in range(k))
        if form == "compound":
            c = rng.choice(["a_ngo", "so_ngo", "a_lo", "multi"])
            if c == "a_ngo":
                return f"{n(200)} {rng.choice(['ngõ', 'ngách'])} {n(700)}"
            if c == "so_ngo":
                return f"{rng.choice(['Số', 'số'])} {n(400)} ngõ {n(500)}"
            if c == "a_lo":
                return f"{n(30)} lô {n(20)}{rng.choice('abc')}"
            return " ".join(n(120) for _ in range(rng.choice([2, 3])))
        return f"{n(500)}/{n(120)}" + (f"/{n(40)}" if rng.random() < 0.3 else "")

    def street(self, rec: dict) -> str:
        rng = self.rng
        s = rec["street"]
        if re.match(r"^(đường|phố|đại lộ|quốc lộ|tỉnh lộ|hương lộ|cầu|đ\.)\s", s, re.IGNORECASE):
            # "Đường Số 11" không bỏ tiền tố: "số 11" đứng một mình là số nhà (L6) theo prompt
            if s.startswith("Đường ") and not s.startswith("Đường Số") and rng.random() < 0.45:
                return s[len("Đường "):]
            return s
        table = [("", 0.55), ("đường ", 0.22), ("Đường ", 0.10), ("Đ. ", 0.04), ("đ. ", 0.02)]
        if rec["province"] == "Hà Nội":
            table.append(("phố ", 0.10))
        return pick(rng, table) + s

    def ward(self, rec: dict) -> str:
        rng = self.rng
        base = rec["ward"]
        numeric = False
        if rec["province"] == "TP.HCM" and (rec.get("district") or "").startswith("Quận") and rng.random() < 0.35:
            base, numeric = str(rng.randint(1, 16)), True
        if rec["ward_type"] == "Xã":
            table = [("Xã ", 0.60), ("X. ", 0.07), ("", 0.33)]
        else:
            table = [("Phường ", 0.55), ("P. ", 0.10), ("P.", 0.03), ("", 0.32)]
        if numeric:
            table = [t for t in table if t[0]]
        return pick(rng, table) + base

    def district(self, rec: dict) -> str | None:
        d = rec.get("district")
        if not d:
            return None
        rng = self.rng
        m = re.match(r"^(Quận|Huyện|Thành phố|Thị xã)\s+(.+)$", d)
        if not m:
            return d
        typ, name = m.groups()
        abbr = rng.choice({"Quận": ["Q. ", "Q.", "Q "], "Huyện": ["H. "], "Thành phố": ["TP. ", "Tp "],
                           "Thị xã": ["TX. "]}[typ]) + name
        if name.isdigit():                      # "Quận 1" không bao giờ gõ trần thành "1"
            return abbr if typ == "Quận" and rng.random() < 0.20 else d
        return pick(rng, [(d, 0.60), (name, 0.30), (abbr, 0.10)])

    def province(self, rec: dict) -> str:
        base = rec["province"]
        variants = {
            "TP.HCM": [("Thành phố Hồ Chí Minh", 0.22), ("Hồ Chí Minh", 0.30), ("TP. Hồ Chí Minh", 0.08),
                       ("TP.HCM", 0.10), ("HCM", 0.06), ("TPHCM", 0.04), ("Sài Gòn", 0.05),
                       ("Thành Phố Hồ Chí Minh", 0.05), ("TP Hồ Chí Minh", 0.05), ("Tp.HCM", 0.05)],
            "Hà Nội": [("Hà Nội", 0.55), ("Thành phố Hà Nội", 0.15), ("TP. Hà Nội", 0.12), ("HN", 0.08),
                       ("TP Hà Nội", 0.05), ("Thủ đô Hà Nội", 0.05)],
            "Đà Nẵng": [("Đà Nẵng", 0.60), ("Thành phố Đà Nẵng", 0.20), ("TP. Đà Nẵng", 0.10), ("ĐN", 0.05),
                        ("Tp Đà Nẵng", 0.05)],
        }.get(base)
        if variants is None:
            if base in ("Cần Thơ", "Hải Phòng", "Huế"):
                variants = [(base, 0.60), ("Thành phố " + base, 0.30), ("TP. " + base, 0.10)]
            else:
                variants = [(base, 0.65), ("Tỉnh " + base, 0.35)]
        return pick(self.rng, variants)

    # -------------------------------------------------------------- dựng câu đầy đủ
    def compose(self, rec: dict) -> tuple[str, list[dict], list[str]]:
        rng = self.rng
        include = list(pick(rng, STRUCTURES))
        if not rec.get("district") and "L3" in include:
            include.remove("L3")
        has_poi = rng.random() < POI_RATIO
        if has_poi:
            r = rng.random()
            if r < 0.12:
                include = [lv for lv in include if lv not in ("L6", "L5")]
            elif r < 0.55:
                include = [lv for lv in include if lv != "L6"]
        if "L2" in include and rng.random() < COUNTRY_RATIO:
            include.append("L1")
        if not include and not has_poi:
            include = ["L6", "L5"]

        surf = {
            "L6": lambda: self.house(rec), "L5": lambda: self.street(rec), "L4": lambda: self.ward(rec),
            "L3": lambda: self.district(rec), "L2": lambda: self.province(rec),
            "L1": lambda: pick(rng, [("Việt Nam", 0.6), ("Viet Nam", 0.1), ("Vietnam", 0.1), ("VN", 0.15),
                                     ("vn", 0.05)]),
        }
        order = [lv for lv in ["L6", "L5", "L4", "L3", "L2", "L1"] if lv in include]
        if has_poi:
            if "L5" in order and rng.random() < POI_AFTER_STREET:
                order.insert(order.index("L5") + 1, "L7")
            else:
                order.insert(0, "L7")
        parts = []
        for lv in order:
            s = self.poi() if lv == "L7" else surf[lv]()
            if s:
                parts.append((lv, self.T.nfc(s)))

        seps = []
        for (a, _), (b, _) in zip(parts, parts[1:]):
            if a == "L6" and b == "L5":
                seps.append(", " if rng.random() < 0.05 else " ")
            elif a == "L7" and b in ("L6", "L5"):
                seps.append(", " if rng.random() < 0.55 else " ")
            else:
                seps.append(", ")
        tags = []
        mode = pick(rng, SEP_MODES)
        n_comma = sum(s == ", " for s in seps)
        if mode != "comma" and n_comma:
            drop = [i for i, s in enumerate(seps) if s == ", " and (mode == "none" or rng.random() < 0.5)]
            for i in drop:
                seps[i] = " "
            if drop:
                tags.append(f"drop_sep:{len(drop)}/{n_comma}")
        seps = ["," if s == ", " and rng.random() < 0.03 else s for s in seps]

        text, spans, pos = "", [], 0
        for i, (lv, s) in enumerate(parts):
            if i:
                text += seps[i - 1]
                pos += len(seps[i - 1])
            spans.append({"level": lv, "start": pos, "end": pos + len(s), "text": s})
            text += s
            pos += len(s)
        return text, spans, tags

    def to_row(self, text: str, spans: list[dict]) -> dict:
        toks = self.T.tokenize(text)
        bio = []
        for _, s, _e in toks:
            sp = next((x for x in spans if x["start"] <= s < x["end"]), None)
            bio.append("O" if sp is None else ("B-" if s == sp["start"] else "I-") + sp["level"])
        return {"text": text, "tokens": [t for t, _, _ in toks], "bio": bio, "trunc": [0] * len(toks),
                "spans": [{**sp, "truncated": False} for sp in spans]}

    # -------------------------------------------------------------- nhiễu hoá
    def abbreviate(self, units: list[dict]) -> str:
        groups, i = [], 0
        while i < len(units):
            lv = units[i]["level"]
            if lv is None or not units[i]["begin"]:
                i += 1
                continue
            j = i + 1
            while j < len(units) and units[j]["level"] == lv and not units[j]["begin"]:
                j += 1
            groups.append((i, j, lv))
            i = j
        self.rng.shuffle(groups)
        for s, e, lv in groups:
            canon = self.norm_key(" ".join(u["text"] for u in units[s:e]))
            forms = [a for a in self.rev_alias.get(canon, ()) if a != canon and len(a) <= len(canon)]
            if lv not in ("L1", "L2"):
                forms = [a for a in forms if " " in a or "." in a or len(a) > 4]
            if not forms:
                continue
            form = self.rng.choice(sorted(forms))
            toks = self.T.tokenize(form) or [(form, 0, len(form))]
            # khoảng cách giữa các token lấy đúng từ chuỗi viết tắt ("đ. số" không thành "đ . số")
            units[s:e] = [{"sep": units[s]["sep"] if k == 0 else form[toks[k - 1][2]:ts], "text": t, "level": lv,
                           "begin": k == 0, "trunc": 0} for k, (t, ts, _) in enumerate(toks)]
            return f"abbrev:{lv}->{form}"
        return "abbrev:noop"

    def augment(self, row: dict) -> dict:
        rng, A = self.rng, self.A
        units = A.to_units(row)
        applied = []
        if rng.random() < AUG["abbreviate"]:
            applied.append(self.abbreviate(units))
        if rng.random() < AUG["strip_diacritics"]:
            applied.append(A.op_strip_diacritics(units, rng))
        if rng.random() < AUG["typo"]:
            applied.append(A.op_typo(units, rng))
        mode = pick(rng, CASING)
        if mode != "keep":
            for u in units:
                t = u["text"]
                u["text"] = t.lower() if mode == "lower" else t.upper() if mode == "upper" else t[:1].upper() + t[1:].lower()
            applied.append("casing:" + mode)
        out = A.from_units(units, row)
        if [t for t, _, _ in self.T.tokenize(out["text"])] != out["tokens"]:
            return {**row, "augment": ["rejected:retokenize_mismatch"]}
        out["augment"] = applied
        return out

    # -------------------------------------------------------------- tiền tố gõ dở
    def choose_cut(self, row: dict) -> int | None:
        text = row["text"]
        n = len(text)
        toks = self.T.tokenize(text)
        span_ends = {sp["end"] for sp in row["spans"] if sp["end"] < n}
        token_end, mid = set(), set()
        for _, s, e in toks:
            sp = next((x for x in row["spans"] if x["start"] <= s < x["end"]), None)
            if sp and e < sp["end"]:
                token_end.add(e)
            mid |= set(range(s + 1, e))
        pools = {"span_end": span_ends, "mid_token": mid, "token_end": token_end}
        pools = {k: sorted(c for c in v if c >= MIN_PREFIX and c < n) for k, v in pools.items()}
        kinds = [(k, w) for k, w in CUT_KINDS if pools[k]]
        if not kinds:
            return None
        cands = pools[pick(self.rng, kinds)]
        weights = [math.exp(-c / CUT_DECAY) + 0.02 for c in cands]
        return self.rng.choices(cands, weights=weights)[0]

    def label_prefix(self, row: dict, cut: int) -> dict | None:
        """Tiền tố text[:cut] + nhãn; `truncated` theo prompt §3.5 (xem docstring đầu file)."""
        text = row["text"]
        p_text = text[:cut].rstrip()
        plen = len(p_text)
        if plen < MIN_PREFIX:
            return None
        toks = [(t, s, e) for t, s, e in self.T.tokenize(text) if s < plen]
        tokens = [p_text[s:min(e, plen)] for _, s, e in toks]
        bio = row["bio"][:len(toks)]
        spans = []
        for sp in row["spans"]:
            if sp["start"] >= plen:
                continue
            end = min(sp["end"], plen)
            spans.append({"level": sp["level"], "start": sp["start"], "end": end,
                          "text": p_text[sp["start"]:end], "truncated": False, "_cut": sp["end"] > plen})
        trunc = [0] * len(toks)
        if spans and spans[-1].pop("_cut"):
            last = spans[-1]
            full_tok, s, e = toks[-1]
            piece = tokens[-1]
            if piece != full_tok:                               # cắt giữa token
                if re.search(r"\d", full_tok):
                    evidence = piece.endswith(("/", "-", "."))
                else:
                    evidence = bool(WORD_RE.search(piece))
            else:                                               # cắt đúng ranh giới token
                in_span = [t for t, ts, _ in toks if ts >= last["start"]]
                words = [self.key(t) for t in in_span if t not in (".", ",")]
                joined = " ".join(w for w in words if w)
                evidence = in_span[-1] in ("-", "/") or (joined != "" and any(
                    ph == joined or ph.startswith(joined + " ") for ph in INDICATORS[last["level"]]))
            if evidence:
                last["truncated"] = True
                trunc[-1] = 1
        for sp in spans:
            sp.pop("_cut", None)
        return {"text": p_text, "tokens": tokens, "bio": bio, "trunc": trunc, "spans": spans}

    # -------------------------------------------------------------- một mẫu
    def sample(self, rec: dict) -> dict | None:
        text, spans, tags = self.compose(rec)
        row = self.augment(self.to_row(text, spans))
        augment = tags + row.pop("augment")
        n_full = len(row["text"])
        if self.rng.random() < FULL_RATIO:
            cut, out = n_full, dict(row)
        else:
            cut = self.choose_cut(row)
            if cut is None:
                return None
            out = self.label_prefix(row, cut)
            if out is None:
                return None
        is_full = cut >= n_full
        out.update({
            "len_nfc": self.T.char_len(out["text"]), "is_full": is_full, "source_id": rec["id"], "cut": cut,
            "augment": augment, "group": self.assign_group(out["spans"]), "n_anchor": self.n_anchor(out["spans"]),
            "is_long": is_full,
        })
        self.check(out)
        return out

    def check(self, r: dict) -> None:
        toks = self.T.tokenize(r["text"])
        assert [t for t, _, _ in toks] == r["tokens"], (r["text"], r["tokens"])
        assert len(r["tokens"]) == len(r["bio"]) == len(r["trunc"]), r
        assert self.is_valid_bio(r["bio"]), (r["text"], r["bio"])
        # spans phải khớp đúng BIO + offset token
        rebuilt, cur = [], None
        for (_, s, e), tag in zip(toks, r["bio"]):
            if tag.startswith("B-"):
                cur = {"level": tag[2:], "start": s, "end": e}
                rebuilt.append(cur)
            elif tag.startswith("I-"):
                cur["end"] = e
            else:
                cur = None
        assert [(x["level"], x["start"], x["end"]) for x in rebuilt] == \
               [(x["level"], x["start"], x["end"]) for x in r["spans"]], (r["text"], r["bio"], r["spans"])
        for i, sp in enumerate(r["spans"]):
            assert r["text"][sp["start"]:sp["end"]] == sp["text"], sp
            assert not sp["truncated"] or i == len(r["spans"]) - 1, r
        assert sum(r["trunc"]) == sum(sp["truncated"] for sp in r["spans"]), r


# ------------------------------------------------------------------ chia & ghi
def split_ids(b: Builder, test_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    held = [r for r in b.records if r.get("street_pool") == "heldout"]
    normal = [r for r in b.records if r.get("street_pool") != "heldout"]
    rng.shuffle(normal)
    n_test = max(0, round(len(b.records) * test_ratio) - len(held))
    return normal[n_test:], held + normal[:n_test]


def generate(b: Builder, recs: list[dict], n: int, seen: set[str], forbid: set[str]) -> list[dict]:
    rows, stall = [], 0
    order = list(recs)
    while len(rows) < n:
        b.rng.shuffle(order)
        before = len(rows)
        for rec in order:
            row = b.sample(rec)
            if row is None:
                continue
            k = b.key(row["text"])
            if row["text"] in seen or k in forbid or k in b.golden_texts:
                continue
            seen.add(row["text"])
            rows.append(row)
            if len(rows) >= n:
                break
        stall = stall + 1 if len(rows) == before else 0
        if stall > 3:
            raise SystemExit(f"Không sinh thêm được mẫu mới (dừng ở {len(rows)}/{n}).")
    return rows


def stats(rows: list[dict]) -> dict:
    full = [r for r in rows if r["is_full"]]
    pref = [r for r in rows if not r["is_full"]]
    lv = Counter(sp["level"] for r in rows for sp in r["spans"])
    has = lambda rs, l: round(sum(any(sp["level"] == l for sp in r["spans"]) for r in rs) / max(1, len(rs)), 3)  # noqa: E731
    lens = sorted(r["len_nfc"] for r in pref) or [0]
    return {
        "rows": len(rows), "full": len(full), "prefix": len(pref), "source_ids": len({r["source_id"] for r in rows}),
        "full_lowercase": round(sum(r["text"] == r["text"].lower() for r in full) / max(1, len(full)), 3),
        "full_no_comma": round(sum("," not in r["text"] for r in full) / max(1, len(full)), 3),
        "full_has_L7": has(full, "L7"),
        "full_avg_len": round(sum(r["len_nfc"] for r in full) / max(1, len(full)), 1),
        "prefix_median_len": lens[len(lens) // 2],
        "prefix_truncated": round(sum(any(sp["truncated"] for sp in r["spans"]) for r in pref) / max(1, len(pref)), 3),
        "prefix_no_span": round(sum(not r["spans"] for r in pref) / max(1, len(pref)), 3),
        "span_levels": dict(sorted(lv.items())),
        "group": dict(Counter(r["group"] for r in rows)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=100_000, help="tổng số mẫu (train + test)")
    ap.add_argument("--test-ratio", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="models/data/phobert_ner")
    ap.add_argument("--parser-repo", default=str(DEFAULT_PARSER_REPO))
    args = ap.parse_args()

    b = Builder(Path(args.parser_repo), args.seed)
    train_recs, test_recs = split_ids(b, args.test_ratio, args.seed)
    n_test = round(args.n * args.test_ratio)
    n_train = args.n - n_test
    print(f"address_db: {len(b.records)} bản ghi dùng được, loại {len(b.excluded_ids)} bản ghi nguồn của golden; "
          f"train ids {len(train_recs)} / test ids {len(test_recs)}")

    seen: set[str] = set()
    train = generate(b, train_recs, n_train, seen, set())
    train_keys = {b.key(r["text"]) for r in train}
    test = generate(b, test_recs, n_test, seen, train_keys)
    assert not {r["source_id"] for r in train} & {r["source_id"] for r in test}, "rò rỉ source_id"

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("test", test)):
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    meta = {
        "seed": args.seed, "n": args.n, "test_ratio": args.test_ratio,
        "parser_repo": str(Path(args.parser_repo).resolve()),
        "labels": ["O"] + [f"{p}-{lv}" for lv in LEVELS for p in ("B", "I")],
        "excluded_golden_source_ids": sorted(b.excluded_ids),
        "config": {"FULL_RATIO": FULL_RATIO, "POI_RATIO": POI_RATIO, "POI_AFTER_STREET": POI_AFTER_STREET,
                   "COUNTRY_RATIO": COUNTRY_RATIO, "SEP_MODES": SEP_MODES, "CASING": CASING, "AUG": AUG,
                   "CUT_KINDS": CUT_KINDS, "CUT_DECAY": CUT_DECAY},
        "stats": {"train": stats(train), "test": stats(test)},
        "golden_reference": {"full_lowercase": 0.62, "full_no_comma": 0.307, "full_has_L7": 0.453,
                             "full_avg_len": 57.8, "prefix_median_len": 16},
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    for name in ("train", "test"):
        print(name, json.dumps(meta["stats"][name], ensure_ascii=False))
    print("golden", json.dumps(meta["golden_reference"], ensure_ascii=False))
    print(f"-> {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
