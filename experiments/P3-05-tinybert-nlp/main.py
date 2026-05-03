from config import Config
from models import create_teacher_model, create_student_model, create_projection_layers
from dataset import get_dataloaders
from train import finetune_teacher, distill_train, train_student_baseline
from evaluate import print_results
from utils import get_device, set_seed


def main():
    config = Config()
    set_seed(config.seed)
    device = get_device()
    print(f"使用设备: {device}")
    print(f"教师模型: {config.teacher_name} ({config.teacher_num_layers}层, {config.teacher_hidden_size}维)")
    print(f"学生模型: TinyBERT ({config.num_student_layers}层, {config.student_hidden_size}维)")
    print(f"蒸馏温度 T={config.T}, α_pred={config.alpha_pred}, α_attn={config.alpha_attn}, α_hidn={config.alpha_hidn}, α_embd={config.alpha_embd}")

    train_loader, val_loader = get_dataloaders(config)

    teacher = create_teacher_model(config).to(device)
    student_kd = create_student_model(config).to(device)
    student_baseline = create_student_model(config).to(device)
    W_e, W_h = create_projection_layers(config)
    W_e = W_e.to(device)
    W_h = W_h.to(device)

    teacher_params = sum(p.numel() for p in teacher.parameters())
    student_params = sum(p.numel() for p in student_kd.parameters())
    print(f"教师参数量: {teacher_params/1e6:.1f}M, 学生参数量: {student_params/1e6:.1f}M")

    print("\n=== 微调教师模型 (BERT-base on SST-2) ===")
    finetune_teacher(teacher, train_loader, val_loader, device, config)

    print("\n=== TinyBERT 三层蒸馏训练 ===")
    distill_train(teacher, student_kd, W_e, W_h, train_loader, val_loader, device, config)

    print("\n=== 直接微调学生模型（基线） ===")
    train_student_baseline(student_baseline, train_loader, val_loader, device, config)

    print_results(teacher, student_kd, student_baseline, val_loader, device)


if __name__ == '__main__':
    main()
