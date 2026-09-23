"""Dich output cua model NER PhoBERT ve schema L1-L7 cua du an.

Nam quy tac chuan hoa, do tren 24 ban ghi output_model.json:
  R1 gop *_TYPE theo nhan       49.3% -> 76.8%
  R4 sua transition IOB2        khong doi (mau qua nho)
  R2 gazetteer quoc gia         -> 79.7%
  R3 gazetteer tinh/thanh       -> 82.6%
  R5 gop tien to theo tu vung   -> 87.0%

KHONG chuan hoa bat dong ngu nghia (vd "so 5" la ten duong hay so nha) —
do la tin hieu ma feedback loop sinh ra de phat hien.
"""
import re, unicodedata

LABELS = ["B_PRO","B_CITY","NUMBER_TYPE","B_DIST","TO_TYPE","B_STREET","I_PRO","I_DIST","PRO_TYPE",
          "OTHER","I_STREET","B_WARD","STREET_TYPE","I_CITY","CITY_TYPE","O","NUMBER","WARD_TYPE",
          "I_WARD","DIST_TYPE","TO"]
TYPE2ENT = {"STREET_TYPE":"STREET","WARD_TYPE":"WARD","DIST_TYPE":"DIST",
            "CITY_TYPE":"CITY","PRO_TYPE":"PRO","NUMBER_TYPE":"NUMBER"}
ENT2L = {"STREET":"L5","WARD":"L4","DIST":"L3","CITY":"L2","PRO":"L2","NUMBER":"L6"}
PREFIX_WORDS = {"thành","phố","tp","tp.","tỉnh","quận","q","q.","huyện","phường","p","p.",
                "xã","x","x.","thị","trấn","đường","số","ngõ","ngách","hẻm","lô","khu"}
COUNTRY = {"viet nam","vietnam","vn"}
PROVINCES = set("""an giang|ba ria vung tau|bac giang|bac kan|bac lieu|bac ninh|ben tre|binh dinh|
binh duong|binh phuoc|binh thuan|ca mau|can tho|cao bang|da nang|dak lak|dak nong|dien bien|dong nai|
dong thap|gia lai|ha giang|ha nam|ha noi|ha tinh|hai duong|hai phong|hau giang|hoa binh|hung yen|
khanh hoa|kien giang|kon tum|lai chau|lam dong|lang son|lao cai|long an|nam dinh|nghe an|ninh binh|
ninh thuan|phu tho|phu yen|quang binh|quang nam|quang ngai|quang ninh|quang tri|soc trang|son la|
tay ninh|thai binh|thai nguyen|thanh hoa|thua thien hue|tien giang|ho chi minh|tra vinh|tuyen quang|
vinh long|vinh phuc|yen bai|sai gon|tphcm|hcm|hn|tp hcm""".replace("\n","").split("|"))
PROVINCES = {p.strip() for p in PROVINCES if p.strip()}
_STRIP_PRE = re.compile(r"^(tp\.?|thanh pho|tinh|t\.)\s+")

def detok(t):  return t[:-2] if t.endswith("@@") else t
def nfc(s):    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", s).lower()).strip()
def fold(s):
    s = unicodedata.normalize("NFD", s.lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn").replace("đ", "d")
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s.]", " ", s)).strip()
def bare(s):   return _STRIP_PRE.sub("", fold(s)).strip()

_PURE_PUNCT = re.compile(r"^[^\w]+$", re.UNICODE)

def force_bpe_continuation(pairs):                              # R6
    """Token ket thuc bang '@@' nghia la KHONG CO KHOANG TRANG truoc token
    sau (de ghep lai dung chuoi goc) — day KHONG dong nghia voi "cung mot
    thuc the". Vd "long@@" dung truoc dau phay "," van co "@@" du day la
    ranh gioi tu, khong phai giua tu.

    Chi ep gop khi token sau la CHU/SO that (khi do @@ moi la bang chung
    dang tin cay ve viec BPE cat doi mot tu, vd "th@@"+"ạnh"="thạnh",
    "W@@"+"er"="Wer"). Bo qua khi token sau la dau cau thuan — vd
    "trang@@" truoc "," khong duoc ep "," thanh mot phan cua thuc the.
    """
    for i in range(len(pairs) - 1):
        tok, lab = pairs[i]
        if not tok.endswith("@@"):
            continue
        nxt_tok = pairs[i + 1][0]
        if _PURE_PUNCT.match(nxt_tok):
            continue
        if lab.startswith(("B_", "I_")) and lab[2:] in ENT2L:
            pairs[i + 1][1] = "I_" + lab[2:]
        elif lab == "NUMBER":
            pairs[i + 1][1] = "NUMBER"
    return pairs

def repair_iob2(pairs):                                        # R4
    prev = None
    for pr in pairs:
        if pr[1].startswith("I_") and (prev is None or prev[2:] != pr[1][2:]):
            pr[1] = "B_" + pr[1][2:]
        prev = pr[1]
    return pairs

def absorb_prefix(pairs):                                      # R5
    for i in range(len(pairs) - 2, -1, -1):
        if detok(pairs[i][0]).lower().rstrip(",") not in PREFIX_WORDS: continue
        nl = pairs[i + 1][1]
        if nl.startswith("B_") and nl[2:] in ENT2L:
            pairs[i][1] = nl; pairs[i + 1][1] = "I_" + nl[2:]
        elif nl.startswith("I_") and nl[2:] in ENT2L and pairs[i][1] != "I_" + nl[2:]:
            pairs[i][1] = "B_" + nl[2:]
    return pairs

def merge_types(pairs):                                        # R1
    out = []
    for i, (t, l) in enumerate(pairs):
        if l in TYPE2ENT:
            ent = TYPE2ENT[l]
            out.append([t, "B_" + ent])
            if i + 1 < len(pairs) and pairs[i + 1][1] == "B_" + ent:
                pairs[i + 1][1] = "I_" + ent
        else:
            out.append([t, l])
    return out

def to_spans(pairs):
    res, cur = [], None
    for t, l in pairs:
        if l.startswith("B_") and l[2:] in ENT2L:
            if cur: res.append(cur)
            cur = [ENT2L[l[2:]], [t]]
        elif l.startswith("I_") and l[2:] in ENT2L and cur and cur[0] == ENT2L[l[2:]]:
            cur[1].append(t)
        elif l == "NUMBER":
            if cur and cur[0] == "L6": cur[1].append(t)
            else:
                if cur: res.append(cur)
                cur = ["L6", [t]]
        else:
            if cur: res.append(cur); cur = None
    if cur: res.append(cur)
    return [[lv, nfc("".join(detok(x) if x.endswith("@@") else detok(x) + " " for x in ts))]
            for lv, ts in res]

def apply_gazetteer(spans):                                    # R2 + R3
    for sp in spans:
        if fold(sp[1]) in COUNTRY: sp[0] = "L1"
        elif sp[0] in ("L3", "L4") and bare(sp[1]) in PROVINCES: sp[0] = "L2"
    return spans

def normalize(token_label_pairs):
    """(token, nhan PhoBERT)[] -> [(level L1-L7, text)] theo schema du an."""
    p = [list(x) for x in token_label_pairs]
    p = force_bpe_continuation(p)   # R6 truoc R4: sua loi tach tu BPE truoc khi kiem IOB2
    p = repair_iob2(p)              # R4
    p = absorb_prefix(p)            # R5
    p = merge_types(p)              # R1
    return apply_gazetteer(to_spans(p))
