"""Kiểm tra hình thức bằng Python thuần, không gọi LLM.

Module sẽ kiểm offset, BIO và enum, đồng thời phân biệt vi phạm ``hard`` chắc
chắn sai với vi phạm ``soft`` đáng ngờ. Validator chạy trước Lớp 1 và chạy lại
sau output của Lớp 3; nó không đánh giá đúng sai ngữ nghĩa sâu.

Chưa implement. Xem BUILD.md §7 để biết thứ tự triển khai.
"""
