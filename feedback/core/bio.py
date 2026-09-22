"""Chuyển đổi hai chiều giữa chuỗi BIO và danh sách span.

Module chỉ chấp nhận tag hệ L và sẽ kiểm tra transition IOB2. ``I-Lx`` sau
``O`` hoặc sau ``B-Ly``/``I-Ly`` khác loại là không hợp lệ; module không gọi
LLM và không tự sửa nhãn ngữ nghĩa.

Chưa implement. Xem BUILD.md §7 để biết thứ tự triển khai.
"""
