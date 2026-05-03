import os
import copy
import torch
import torch.nn.functional as F

from config import Config
from models import SimpleUNet, DiffusionSampler
from losses import DiffusionMSELoss, DistillMSELoss


def train_ddpm(model, sampler, train_loader, device, config=Config()):
    criterion = DiffusionMSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    sampler = sampler.to(device)

    for epoch in range(config.teacher_epochs):
        model.train()
        total_loss = 0
        for batch_idx, (images, _) in enumerate(train_loader):
            images = images.to(device)
            batch_size = images.shape[0]
            t = torch.randint(0, config.num_timesteps, (batch_size,), device=device)

            x_t, noise = sampler.add_noise(images, t)
            noise_pred = model(x_t, t)
            loss = criterion(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f'[教师DDPM] Epoch {epoch+1}/{config.teacher_epochs}, Loss: {avg_loss:.6f}')

    return model


def _get_student_timesteps(num_timesteps, student_steps):
    step_size = num_timesteps // student_steps
    timesteps = list(range(0, num_timesteps, step_size))
    return timesteps


@torch.no_grad()
def _teacher_two_step(teacher, sampler, x_t, t_cur, t_mid, t_prev):
    eps_pred_1 = teacher(x_t, t_cur)
    alpha_t = sampler.alphas_cumprod[t_cur]
    alpha_mid = sampler.alphas_cumprod[t_mid]
    x0_pred_1 = (x_t - torch.sqrt(1 - alpha_t) * eps_pred_1) / torch.sqrt(alpha_t)
    x0_pred_1 = torch.clamp(x0_pred_1, -1.0, 1.0)
    dir_xt = torch.sqrt(1 - alpha_mid) * eps_pred_1
    x_mid = torch.sqrt(alpha_mid) * x0_pred_1 + dir_xt

    eps_pred_2 = teacher(x_mid, t_mid)
    alpha_mid_val = sampler.alphas_cumprod[t_mid]
    alpha_prev = sampler.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=x_t.device)
    x0_pred_2 = (x_mid - torch.sqrt(1 - alpha_mid_val) * eps_pred_2) / torch.sqrt(alpha_mid_val)
    x0_pred_2 = torch.clamp(x0_pred_2, -1.0, 1.0)

    return x0_pred_2


def _student_one_step(student, sampler, x_t, t_cur, t_prev):
    eps_pred = student(x_t, t_cur)
    alpha_t = sampler.alphas_cumprod[t_cur]
    alpha_prev = sampler.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=x_t.device)
    x0_pred = (x_t - torch.sqrt(1 - alpha_t) * eps_pred) / torch.sqrt(alpha_t)
    x0_pred = torch.clamp(x0_pred, -1.0, 1.0)
    return x0_pred


def progressive_distill(teacher, sampler, train_loader, device, config=Config()):
    sampler = sampler.to(device)
    criterion = DistillMSELoss()
    results = {}

    current_model = copy.deepcopy(teacher)
    current_steps = config.step_schedule[0]

    for stage_idx in range(1, len(config.step_schedule)):
        target_steps = config.step_schedule[stage_idx]
        print(f'\n{"="*50}')
        print(f'渐进蒸馏阶段 {stage_idx}: {current_steps}步 → {target_steps}步')
        print(f'{"="*50}')

        student = copy.deepcopy(current_model)
        student.train()
        teacher_stage = current_model
        teacher_stage.eval()

        optimizer = torch.optim.Adam(student.parameters(), lr=config.learning_rate)

        step_size_teacher = config.num_timesteps // current_steps
        step_size_student = config.num_timesteps // target_steps

        teacher_timesteps = list(range(0, config.num_timesteps, step_size_teacher))

        for epoch in range(config.distill_epochs_per_stage):
            student.train()
            total_loss = 0
            for images, _ in train_loader:
                images = images.to(device)
                batch_size = images.shape[0]

                pair_idx = torch.randint(0, len(teacher_timesteps) - 1, (batch_size,), device=device)
                t_cur_idx = pair_idx
                t_mid_idx = pair_idx + 1

                t_cur = torch.tensor([teacher_timesteps[i] for i in t_cur_idx], device=device, dtype=torch.long)
                t_mid = torch.tensor([teacher_timesteps[i] for i in t_mid_idx], device=device, dtype=torch.long)

                t_prev_val = 2 * t_mid - t_cur
                t_prev = torch.clamp(t_prev_val, min=0, max=config.num_timesteps - 1)

                x_t, _ = sampler.add_noise(images, t_cur)

                with torch.no_grad():
                    x0_teacher = _teacher_two_step(teacher_stage, sampler, x_t, t_cur, t_mid, t_prev)

                x0_student = _student_one_step(student, sampler, x_t, t_cur, t_prev)

                loss = criterion(x0_student, x0_teacher)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f'[蒸馏 {current_steps}→{target_steps}] Epoch {epoch+1}/{config.distill_epochs_per_stage}, Loss: {avg_loss:.6f}')

        current_model = student
        current_steps = target_steps
        results[target_steps] = copy.deepcopy(student.state_dict())

        save_path = os.path.join(config.save_dir, f'student_{target_steps}steps.pt')
        os.makedirs(config.save_dir, exist_ok=True)
        torch.save(student.state_dict(), save_path)
        print(f'已保存 {target_steps}步模型到 {save_path}')

    return current_model, results
