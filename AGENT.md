# Agent Instructions

## 1. Phạm vi làm việc (Scope)
- Chỉ được đọc, chỉnh sửa, tạo hoặc xoá file **bên trong thư mục dự án hiện tại** (project root và các thư mục con của nó).
- Không tự ý truy cập, tham chiếu hoặc thao tác với file/thư mục nằm ngoài phạm vi dự án (ví dụ: đi ra ngoài qua `../`, truy cập home directory, thư mục hệ thống, hoặc các dự án khác trên máy).
- Nếu một tác vụ yêu cầu truy cập ra ngoài thư mục dự án (ví dụ: đọc config toàn cục, dependency ở nơi khác), **phải dừng lại và hỏi người dùng xác nhận trước**, nêu rõ đường dẫn cụ thể cần truy cập và lý do.

## 2. Xử lý hành động không chắc chắn (Confidence threshold)
- Trước khi thực hiện bất kỳ hành động nào có khả năng ảnh hưởng đến code, dữ liệu hoặc cấu trúc dự án, tự đánh giá mức độ tự tin (confidence) về tính đúng đắn của hành động đó.
- Nếu confidence nằm trong khoảng **50% - 60%** (tức là không chắc chắn, có rủi ro sai hoặc gây tác động không mong muốn):
  - **Không tự ý thực hiện.**
  - Dừng lại, trình bày rõ:
    - Hành động dự định làm
    - Lý do không chắc chắn / các rủi ro có thể xảy ra
    - Các phương án thay thế (nếu có)
  - Chờ người dùng xác nhận hoặc chọn phương án trước khi tiếp tục.
- Nếu confidence dưới 50%: coi như chưa đủ thông tin, cần hỏi thêm để làm rõ trước khi đề xuất bất kỳ hành động nào.
- Nếu confidence trên 60%: có thể tiến hành, nhưng vẫn nên báo cáo ngắn gọn hành động đã/đang thực hiện.