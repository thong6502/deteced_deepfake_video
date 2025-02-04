import torch
import torch.nn as nn
from .config import MODEL, MODEL_WEIGHTS
from peft import LoraConfig, get_peft_model


# Hàm để lấy tên của tất cả các lớp Linear trong mô hình
def get_linear_layer_names(model):
    linear_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):  # Kiểm tra nếu module là lớp Linear
            linear_layers.append(name)  # Thêm tên lớp vào danh sách
    return linear_layers


# Định nghĩa mô hình
class MyConvNeXt(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        # Tải mô hình ConvNeXt Tiny với weights mặc định
        self.model = MODEL(weights=MODEL_WEIGHTS)
        
        # Thay đổi lớp classifier cuối cùng để phù hợp với số lớp đầu ra
        self.model.classifier[2] = nn.Linear(self.model.classifier[2].in_features, num_classes)

        # Lấy tên của tất cả các lớp Linear trong mô hình
        target_modules = get_linear_layer_names(self.model)
        # print("Các lớp Linear trong mô hình:", target_modules)

        # Áp dụng LoRA cho tất cả các lớp Linear
        config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=target_modules,  # Áp dụng LoRA cho tất cả các lớp Linear
            lora_dropout=0.1,
        )
        self.model = get_peft_model(self.model, config)

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    # Khởi tạo mô hình với số lớp đầu ra là 2
    model = MyConvNeXt(num_classes=2)
    
    # Di chuyển mô hình lên GPU nếu có
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # In thông số có thể huấn luyện
    model.model.print_trainable_parameters()

    