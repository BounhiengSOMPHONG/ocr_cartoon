# manga-page-ocr — Full-page English manga OCR (standalone)

แพ็กเกจแยกออกมาจากโปรเจกต์ `manga-translator` (2026-10-05) สำหรับเอาไปใช้ใน
โปรเจกต์อื่น: **รับภาพหน้าเต็ม → คืนกล่องข้อความ + ข้อความ** (เส้นทาง
อังกฤษ/ละติน) โดยคงสูตร inference เดิมของโปรเจกต์ไว้ทุกค่า

```
image ──► DBNet (detect-20241225) ──► MIT48px beam OCR ──► CTC fallback + lexicon gate
                                                                      │
                          TextRegion[] (bbox, text, confidence, flag) ◄─┘ + panel-aware reading order
```

## ติดตั้ง

```bash
pip install -r requirements.txt
```

- **CUDA**: ติดตั้ง `torch`/`torchvision` เวอร์ชันที่ตรงกับเครื่องจาก
  https://pytorch.org ก่อน แล้วค่อย `pip install -r requirements.txt`
  (ของโปรเจกต์ต้นทางทดสอบกับ torch 2.13 / torchvision คู่กัน บน RTX 2050 4GB)
- ไม่มี GPU ก็ได้: ส่ง `device='cpu'` (ช้ากว่ามาก)

## ใช้

```bash
python example.py page.jpg                  # พิมพ์กล่อง + ข้อความ
python example.py page.jpg --out page.jsonl # เขียน JSONL ด้วย
python example.py page.jpg --device cpu --rtl
```

```python
from manga_page_ocr import PageOCR, load_image_rgb, save_jsonl

ocr = PageOCR()                                    # models/ ข้าง ๆ แพ็กเกจ
regions = ocr.run_sync(load_image_rgb('page.jpg'), 'page.jpg')
save_jsonl(regions, 'page.jsonl')

# ในแอป async: โหลดครั้งเดียวแล้ว reuse ข้ามหน้า
await ocr.load()
regions = await ocr.run(image_rgb, 'page.jpg')
```

ปรับค่าได้ผ่าน `PageOCR(options={...})` — ดู `DEFAULT_OPTIONS` ใน `page.py`
(ค่า default = ค่าที่โปรเจกต์ต้นทางใช้วัดผลจริง):

| กลุ่ม | ค่า | ที่มา |
|---|---|---|
| `detector` | `detect_size 2048`, `text_threshold 0.5`, `box_threshold 0.7`, `unclip_ratio 2.3` | reference defaults |
| `ocr` | `text_height 32`, `beams_k 5`, `padding_pct 5.0`, `src_scale 1.0`, `binarize none` | exp-002 adopt / exp-003,004 reject |
| `fallback` | `enabled`, `gate_min_prob 0.75`, `disagreement true`, `ctc_min_prob 0.5` | Phase 3 + exp-105/117 |
| `reading_order` | `right_to_left` (false = อังกฤษ) | Phase 4 (Kumiko panels) |

> `beams_k` อย่าลดโดยไม่มีหน้าจริงทดสอบ: exp-109 ลดเป็น 2 แล้วเร็วขึ้น 5.9x
> แต่ exp-110 ถอยกลับเพราะข้อความบนหน้าผู้ใช้เพี้ยน

## Output schema (`TextRegion`)

หนึ่งบรรทัด JSON ต่อหนึ่งกล่อง — ฟิลด์สำคัญ:

| ฟิลด์ | ความหมาย |
|---|---|
| `bbox`, `polygon` | กรอบ axis-aligned + 4 จุดของกล่อง |
| `text` | ข้อความที่ยอมรับเป็นผลสุดท้าย (beam + gate + lexicon แล้ว) |
| `confidence` | log-prob เฉลี่ยของ backend ที่ชนะ |
| `direction` | `horizontal` / `vertical` (แนวตั้งถูกหมุนก่อนอ่าน) |
| `flag` | `None` = ผ่าน / `low_confidence` = ต่ำกว่า gate / `needs_review` = gate กับ CTC ไม่ลงรอยหรือได้ control token |
| `detection_score` | คะแนน detector (ไม่ถูก OCR ทับ) |
| `ocr`, `primary_text`, `primary_confidence` | provenance: backend ที่ให้ผลสุดท้าย + ผลตั้งต้นก่อน gate |
| `panel_index` | ลำดับ panel ตามการอ่าน (Phase 4) |
| `fg_color`, `bg_color` | สีตัวอักษร/พื้นหลังที่โมเดลทำนาย (0-255) |

แนะนำให้โปรเจกต์ปลายทาง **กรอง `flag is None`** ก่อนใช้ข้อความอัตโนมัติ
แล้วเอาที่ flagged ไปให้คนตรวจ

## โมเดล (`models/`, ~620MB — ต้องมีครบ)

| ไฟล์ | ใช้ทำอะไร |
|---|---|
| `detect-20241225.ckpt` (294MB) | DBNet text detector |
| `ocr/ocr.ckpt` (163MB) + `ocr/alphabet-all-v5.txt` | MIT48px beam recognizer |
| `ocr-ctc/ocr-ctc.ckpt` (161MB) + `ocr-ctc/alphabet-all-v5.txt` | CTC fallback (gate) |

โหลดใหม่ได้จาก release `beta-0.3` ของ
[zyddnys/manga-image-translator](https://github.com/zyddnys/manga-image-translator/releases/tag/beta-0.3)
(`detect-20241225.ckpt`, `ocr.zip`, `ocr-ctc.zip`) แล้ววางให้ตรงโครงสร้างข้างบน
(ใน `models/` ต้องมี `ocr/` และ `ocr-ctc/` เป็นโฟลเดอร์)

## ความถูกต้อง (วัดแล้ว 2026-10-05 กับโปรเจกต์ต้นทาง)

รันจากนอกโปรเจกต์ (cwd = `/tmp`) แล้วเทียบกับผลที่โปรเจกต์ต้นทางบันทึกไว้:

- `testmanga1.jpg` (หน้าที่จูน): **29/29 กล่อง ตรงกันทุกฟิลด์** กับ
  `results/exp-117-en.jsonl` (รวม `primary_*`, `ocr` chain)
- `en-blondie-1930.png` (held-out, คอนทราสต์ต่ำ): **45/45 กล่อง ผลสุดท้าย
  (`text`/`bbox`/`flag`/`confidence`/`detection_score`) ตรงกันทั้งหมด** กับ
  `results/exp-117-heldout.jsonl` — ต่างเฉพาะป้าย `ocr` chain กับ `primary_*`
  เพราะไฟล์อ้างอิงรัน backend แบบ auto (JA+EN) ส่วนแพ็กเกจนี้เป็น EN-only

## ข้อจำกัด

- **EN/Latin เท่านั้น** — ไม่มี manga-ocr (ญี่ปุ่น) และไม่มี per-box script
  routing ของโหมด auto; หน้าที่เป็นญี่ปุ่นจะได้ผลไม่ถูกต้อง
- ไม่มี crops / visualization / overlay / inpainting ของโปรเจกต์ต้นทาง —
  ได้ JSONL กับ API อย่างเดียว (โครงสร้างอยู่ใน `page.py` เผื่ออยากเพิ่ม)
- `torchvision` dep มาจาก backbone ของ DBNet; `shapely`/`py3langid`/`tqdm`/
  `requests`/`colorama` มาจากโค้ด vendored (แม้บางตัวไม่ได้ถูกเรียกในเส้นทางนี้)

## ลิขสิทธิ์ / ที่มา

- **โค้ดใน `manga_page_ocr/vendor/`** (detection, ocr, utils) คัดลอกจาก
  [zyddnys/manga-image-translator](https://github.com/zyddnys/manga-image-translator)
  ซึ่งเป็น **GPL-3.0** → ถ้าโปรเจกต์ปลายทางแจกจ่าย ต้องเปิดซอร์สตามเงื่อนไข GPL
- **โมเดล** (checkpoint) มาจาก release เดียวกัน (beta-0.3)
- `manga_page_ocr/data/en-word-frequency.tsv` — สกัดจาก wordfreq 3.1.1
  (ที่มา/เงื่อนไขละเอียดใน `manga_page_ocr/data/ATTRIBUTION.md`)
- โค้ดที่เหลือ (`page.py`, `recognize.py`, `detect.py`, `validity.py`,
  `lexicon.py`, `reading_order.py`, `schema.py`) มาจากโปรเจกต์ต้นทาง
  `manga-translator` เช่นกัน
