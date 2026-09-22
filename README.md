# Feedback loop cho NER địa chỉ tiếng Việt

Subsystem `feedback/` chạy sau pipeline NER L1–L7 để phát hiện parse sai, đề xuất bản sửa, so sánh các phương án và đo xem vòng sửa có thực sự cải thiện chất lượng dữ liệu hay không.

| Tầng | Công cụ | Vai trò | Tối ưu cho |
|---|---|---|---|
| Lớp 0 | Python thuần, không LLM | Bắt lỗi hình thức (offset, BIO, enum) | Chi phí 0 |
| Lớp 1 | SLM | Detector — phát hiện nghi ngờ và parse lại | Recall cao |
| Lớp 2 | DeepSeek | Comparator — chấm điểm parse cũ và mới | Precision |
| Lớp 3 | OpenAI | Arbiter — chốt, viết lại hoặc đẩy cho người | Đúng tuyệt đối |

## Cài đặt

```bash
python -m venv .venv
pip install -r requirements.txt
```

Sao chép `.env.example` thành `.env` và điền cấu hình model cần dùng. Thứ tự triển khai và tiêu chí nghiệm thu nằm trong [BUILD.md](BUILD.md).
