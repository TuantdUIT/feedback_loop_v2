"""Lớp bọc chung để gọi 3 nhà cung cấp LLM.

Module sẽ retry lỗi tạm thời, ép output theo JSON schema và ghi lại token cùng
chi phí cho mỗi lần gọi. Nó không chứa prompt nhiệm vụ hay logic gate của
cascade.

Chưa implement. Xem BUILD.md §7 để biết thứ tự triển khai.
"""
