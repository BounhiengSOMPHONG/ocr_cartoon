# ที่มาของไฟล์ในโฟลเดอร์นี้

## `en-word-frequency.tsv` — ตารางความถี่คำอังกฤษ

- **ที่มา**: ไลบรารี [wordfreq](https://pypi.org/project/wordfreq/) 3.1.1
  (`zipf_frequency` + `top_n_list('en', 300000, ascii_only=True)`)
- **ลิขสิทธิ์**: ตัวโค้ด wordfreq = Apache-2.0; ตัว*ข้อมูล*ความถี่รวบรวมจาก
  หลายคลัง (OpenSubtitles, Wikipedia, subtlex ฯลฯ) แต่ละคลังมีเงื่อนไขของตัวเอง
  — ไฟล์นี้เป็น**ตัวเลขความถี่**ที่สกัดแล้ว ไม่ใช่ตัวข้อความจากคลังเหล่านั้น
- **สร้างเมื่อ**: 2026-09-30 · 89,630 บรรทัด รูปแบบ `<คำ>\t<zipf>`
  (คัดเฉพาะ a-z, ยาว ≥ 2, zipf ≥ 2.0)
- **ใช้ทำอะไร**: `manga_page_ocr/lexicon.py` ใช้ตัดสินว่าระหว่างสองข้อความที่
  recognizer อ่านได้ อันไหน "เป็นคำอังกฤษจริง" (origin project: exp-117) —
  แพ็กเกจจึงไม่ต้องพึ่ง wordfreq ตอนรัน (vendored แบบ static)
