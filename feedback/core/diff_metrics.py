"""So sánh hai bản parse và phân loại từng điểm khác biệt.

Module sẽ phân loại COR/INC/PAR/MIS/SPU, tách lỗi biên khỏi lỗi nhãn và xem
``truncated`` như chiều đúng/sai thứ ba. Vì spans rỗng có thể là đáp án đúng,
module cũng cần metric abstain riêng và không tự quyết gate.

Chưa implement. Xem BUILD.md §7 để biết thứ tự triển khai.
"""
