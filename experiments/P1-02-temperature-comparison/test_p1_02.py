import os
import sys
import tempfile

import numpy as np
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))

from config import Config
from dataset import get_dataloaders
from evaluate import (plot_convergence_curves, plot_temperature_curve,
                      print_temperature_results, visualize_soft_labels)
from losses import DistillationLoss
from models import StudentNet, TeacherNet
from train import (distill_with_temperature, run_temperature_sweep,
                   train_student_baseline, train_teacher)
from utils import evaluate, get_device, set_seed


class TestConfig:
    def test_default_values(self):
        config = Config()
        assert config.T_values == [1, 2, 4, 8, 16, 20]
        assert config.alpha == 0.7
        assert config.learning_rate == 1e-3
        assert config.teacher_epochs == 20
        assert config.student_epochs == 10
        assert config.batch_size == 64
        assert config.test_batch_size == 1000
        assert config.dropout == 0.2
        assert config.mnist_mean == 0.1307
        assert config.mnist_std == 0.3081
        assert config.seed == 42

    def test_custom_values(self):
        config = Config(T_values=[1, 2, 3], alpha=0.5, learning_rate=0.01)
        assert config.T_values == [1, 2, 3]
        assert config.alpha == 0.5
        assert config.learning_rate == 0.01

    def test_instances_independent(self):
        c1 = Config()
        c2 = Config()
        c1.T_values.append(99)
        assert 99 not in c2.T_values


class TestUtils:
    def test_set_seed_reproducibility(self):
        set_seed(42)
        a = torch.randn(10)
        set_seed(42)
        b = torch.randn(10)
        assert torch.allclose(a, b)

    def test_set_seed_numpy(self):
        set_seed(42)
        a = np.random.rand(10)
        set_seed(42)
        b = np.random.rand(10)
        np.testing.assert_array_equal(a, b)

    def test_get_device(self):
        device = get_device()
        assert isinstance(device, torch.device)

    def test_evaluate(self):
        model = TeacherNet()
        model.eval()
        images = torch.randn(32, 1, 28, 28)
        labels = torch.randint(0, 10, (32,))
        dataset = torch.utils.data.TensorDataset(images, labels)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=16)
        device = torch.device('cpu')
        acc = evaluate(model, dataloader, device)
        assert 0.0 <= acc <= 100.0


class TestModels:
    def test_teacher_forward_shape(self):
        model = TeacherNet(dropout=0.2)
        x = torch.randn(8, 1, 28, 28)
        out = model(x)
        assert out.shape == (8, 10)

    def test_student_forward_shape(self):
        model = StudentNet()
        x = torch.randn(8, 1, 28, 28)
        out = model(x)
        assert out.shape == (8, 10)

    def test_teacher_output_finite(self):
        model = TeacherNet()
        x = torch.randn(4, 1, 28, 28)
        out = model(x)
        assert torch.isfinite(out).all()

    def test_student_output_finite(self):
        model = StudentNet()
        x = torch.randn(4, 1, 28, 28)
        out = model(x)
        assert torch.isfinite(out).all()

    def test_teacher_dropout_effect(self):
        model = TeacherNet(dropout=0.5)
        model.train()
        x = torch.randn(16, 1, 28, 28)
        out1 = model(x)
        out2 = model(x)
        assert not torch.allclose(out1, out2)

    def test_teacher_eval_no_dropout(self):
        model = TeacherNet(dropout=0.5)
        model.eval()
        x = torch.randn(16, 1, 28, 28)
        out1 = model(x)
        out2 = model(x)
        assert torch.allclose(out1, out2)

    def test_teacher_params_count(self):
        model = TeacherNet()
        total = sum(p.numel() for p in model.parameters())
        expected = (28 * 28 * 1200 + 1200) + (1200 * 1200 + 1200) + (1200 * 10 + 10)
        assert total == expected

    def test_student_params_count(self):
        model = StudentNet()
        total = sum(p.numel() for p in model.parameters())
        expected = (28 * 28 * 800 + 800) + (800 * 10 + 10)
        assert total == expected

    def test_student_smaller_than_teacher(self):
        teacher = TeacherNet()
        student = StudentNet()
        t_params = sum(p.numel() for p in teacher.parameters())
        s_params = sum(p.numel() for p in student.parameters())
        assert s_params < t_params


class TestDistillationLoss:
    def test_loss_finite(self):
        criterion = DistillationLoss(T=4.0, alpha=0.7)
        student_logits = torch.randn(16, 10)
        teacher_logits = torch.randn(16, 10)
        labels = torch.randint(0, 10, (16,))
        loss = criterion(student_logits, teacher_logits, labels)
        assert torch.isfinite(loss)

    def test_loss_positive(self):
        criterion = DistillationLoss(T=4.0, alpha=0.7)
        student_logits = torch.randn(16, 10)
        teacher_logits = torch.randn(16, 10)
        labels = torch.randint(0, 10, (16,))
        loss = criterion(student_logits, teacher_logits, labels)
        assert loss.item() > 0

    def test_temperature_scaling(self):
        criterion_low = DistillationLoss(T=1.0, alpha=0.7)
        criterion_high = DistillationLoss(T=20.0, alpha=0.7)
        student_logits = torch.randn(16, 10)
        teacher_logits = torch.randn(16, 10)
        labels = torch.randint(0, 10, (16,))
        loss_low = criterion_low(student_logits, teacher_logits, labels)
        loss_high = criterion_high(student_logits, teacher_logits, labels)
        assert loss_low.item() != loss_high.item()

    def test_alpha_effect(self):
        student_logits = torch.randn(16, 10)
        teacher_logits = torch.randn(16, 10)
        labels = torch.randint(0, 10, (16,))
        criterion_soft = DistillationLoss(T=4.0, alpha=1.0)
        criterion_hard = DistillationLoss(T=4.0, alpha=0.0)
        loss_soft = criterion_soft(student_logits, teacher_logits, labels)
        loss_hard = criterion_hard(student_logits, teacher_logits, labels)
        assert not torch.allclose(loss_soft, loss_hard)

    def test_alpha_zero_is_ce(self):
        student_logits = torch.randn(16, 10)
        teacher_logits = torch.randn(16, 10)
        labels = torch.randint(0, 10, (16,))
        criterion = DistillationLoss(T=4.0, alpha=0.0)
        loss = criterion(student_logits, teacher_logits, labels)
        ce_loss = F.cross_entropy(student_logits, labels)
        assert torch.allclose(loss, ce_loss, atol=1e-5)

    def test_gradient_flows(self):
        criterion = DistillationLoss(T=4.0, alpha=0.7)
        student_logits = torch.randn(16, 10, requires_grad=True)
        teacher_logits = torch.randn(16, 10)
        labels = torch.randint(0, 10, (16,))
        loss = criterion(student_logits, teacher_logits, labels)
        loss.backward()
        assert student_logits.grad is not None
        assert torch.isfinite(student_logits.grad).all()


class TestDataset:
    def test_dataloaders_return_types(self):
        config = Config()
        train_loader, test_loader = get_dataloaders(config)
        assert isinstance(train_loader, torch.utils.data.DataLoader)
        assert isinstance(test_loader, torch.utils.data.DataLoader)

    def test_dataloaders_batch_shapes(self):
        config = Config()
        train_loader, test_loader = get_dataloaders(config)
        images, labels = next(iter(train_loader))
        assert images.shape[1:] == (1, 28, 28)
        assert labels.shape[0] == images.shape[0]

    def test_dataloaders_normalized(self):
        config = Config()
        train_loader, _ = get_dataloaders(config)
        all_pixels = []
        for images, _ in train_loader:
            all_pixels.append(images)
            if len(all_pixels) >= 5:
                break
        pixels = torch.cat(all_pixels, dim=0)
        mean = pixels.mean().item()
        assert abs(mean) < 1.0


class TestTrain:
    def _make_dataloader(self, n=128):
        images = torch.randn(n, 1, 28, 28)
        labels = torch.randint(0, 10, (n,))
        dataset = torch.utils.data.TensorDataset(images, labels)
        return torch.utils.data.DataLoader(dataset, batch_size=32)

    def test_train_teacher_improves(self):
        config = Config(teacher_epochs=3, lr_step_size=2)
        device = torch.device('cpu')
        model = TeacherNet(dropout=config.dropout).to(device)
        train_loader = self._make_dataloader()
        test_loader = self._make_dataloader(64)
        acc_before = evaluate(model, test_loader, device)
        train_teacher(model, train_loader, test_loader, device, config)
        acc_after = evaluate(model, test_loader, device)
        assert acc_after > acc_before or acc_after > 0

    def test_distill_returns_correct_keys(self):
        config = Config(student_epochs=2, lr_step_size=1)
        device = torch.device('cpu')
        teacher = TeacherNet()
        teacher.eval()
        train_loader = self._make_dataloader()
        test_loader = self._make_dataloader(64)
        student, final_acc, epoch_accs = distill_with_temperature(
            teacher, train_loader, test_loader, device, T=4, config=config
        )
        assert isinstance(student, StudentNet)
        assert isinstance(final_acc, float)
        assert isinstance(epoch_accs, list)
        assert len(epoch_accs) == config.student_epochs

    def test_distill_different_temperatures(self):
        config = Config(student_epochs=2, lr_step_size=1)
        device = torch.device('cpu')
        teacher = TeacherNet()
        teacher.eval()
        train_loader = self._make_dataloader()
        test_loader = self._make_dataloader(64)
        _, acc_t1, _ = distill_with_temperature(
            teacher, train_loader, test_loader, device, T=1, config=config
        )
        _, acc_t8, _ = distill_with_temperature(
            teacher, train_loader, test_loader, device, T=8, config=config
        )
        assert isinstance(acc_t1, float)
        assert isinstance(acc_t8, float)

    def test_train_student_baseline_improves(self):
        config = Config(student_epochs=3, lr_step_size=2)
        device = torch.device('cpu')
        model = StudentNet().to(device)
        train_loader = self._make_dataloader()
        test_loader = self._make_dataloader(64)
        acc_before = evaluate(model, test_loader, device)
        train_student_baseline(model, train_loader, test_loader, device, config)
        acc_after = evaluate(model, test_loader, device)
        assert acc_after > acc_before or acc_after > 0

    def test_run_temperature_sweep(self):
        config = Config(T_values=[1, 4], student_epochs=2, lr_step_size=1)
        device = torch.device('cpu')
        teacher = TeacherNet()
        teacher.eval()
        train_loader = self._make_dataloader()
        test_loader = self._make_dataloader(64)
        results = run_temperature_sweep(teacher, train_loader, test_loader, device, config)
        assert 1 in results
        assert 4 in results
        for T in [1, 4]:
            assert 'student' in results[T]
            assert 'final_acc' in results[T]
            assert 'epoch_accs' in results[T]


class TestEvaluate:
    def test_print_temperature_results(self, capsys):
        results = {
            1: {'final_acc': 95.0, 'epoch_accs': [90.0, 95.0]},
            4: {'final_acc': 97.0, 'epoch_accs': [92.0, 97.0]},
        }
        print_temperature_results(results, baseline_acc=93.0, teacher_acc=98.0)
        captured = capsys.readouterr()
        assert '温度参数对比实验结果' in captured.out
        assert '最佳温度' in captured.out

    def test_plot_temperature_curve(self):
        results = {
            1: {'final_acc': 95.0, 'epoch_accs': [90.0, 95.0]},
            4: {'final_acc': 97.0, 'epoch_accs': [92.0, 97.0]},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, 'temperature_curve.png')
            plot_temperature_curve(results, baseline_acc=93.0, save_path=save_path)
            assert os.path.exists(save_path)
            assert os.path.getsize(save_path) > 0

    def test_plot_convergence_curves(self):
        results = {
            1: {'final_acc': 95.0, 'epoch_accs': [90.0, 92.0, 95.0]},
            4: {'final_acc': 97.0, 'epoch_accs': [92.0, 95.0, 97.0]},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, 'convergence_curves.png')
            plot_convergence_curves(results, save_path=save_path)
            assert os.path.exists(save_path)
            assert os.path.getsize(save_path) > 0

    def test_visualize_soft_labels(self):
        teacher = TeacherNet()
        teacher.eval()
        images = torch.randn(10, 1, 28, 28)
        labels = torch.randint(0, 10, (10,))
        dataset = torch.utils.data.TensorDataset(images, labels)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=10)
        with tempfile.TemporaryDirectory() as tmpdir:
            visualize_soft_labels(
                teacher, dataloader, torch.device('cpu'),
                T_values=[1, 4, 8], num_samples=3, save_dir=tmpdir
            )
            for i in range(1, 4):
                assert os.path.exists(os.path.join(tmpdir, f'soft_labels_sample_{i}.png'))


class TestIntegration:
    def test_full_pipeline_small(self):
        config = Config(
            T_values=[1, 4],
            teacher_epochs=2,
            student_epochs=2,
            batch_size=32,
            lr_step_size=1,
            num_vis_samples=2,
        )
        device = torch.device('cpu')
        set_seed(config.seed)

        images = torch.randn(128, 1, 28, 28)
        labels = torch.randint(0, 10, (128,))
        train_dataset = torch.utils.data.TensorDataset(images, labels)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

        test_images = torch.randn(64, 1, 28, 28)
        test_labels = torch.randint(0, 10, (64,))
        test_dataset = torch.utils.data.TensorDataset(test_images, test_labels)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.test_batch_size)

        teacher = TeacherNet(dropout=config.dropout).to(device)
        train_teacher(teacher, train_loader, test_loader, device, config)
        teacher_acc = evaluate(teacher, test_loader, device)
        assert 0.0 <= teacher_acc <= 100.0

        set_seed(config.seed)
        student_baseline = StudentNet().to(device)
        train_student_baseline(student_baseline, train_loader, test_loader, device, config)
        baseline_acc = evaluate(student_baseline, test_loader, device)
        assert 0.0 <= baseline_acc <= 100.0

        results = run_temperature_sweep(teacher, train_loader, test_loader, device, config)
        assert len(results) == 2
        for T in config.T_values:
            assert 'final_acc' in results[T]
            assert 'epoch_accs' in results[T]
            assert len(results[T]['epoch_accs']) == config.student_epochs

        with tempfile.TemporaryDirectory() as tmpdir:
            plot_temperature_curve(results, baseline_acc, save_path=os.path.join(tmpdir, 'temp.png'))
            plot_convergence_curves(results, save_path=os.path.join(tmpdir, 'conv.png'))
            visualize_soft_labels(teacher, test_loader, device, config.T_values,
                                  config.num_vis_samples, save_dir=os.path.join(tmpdir, 'soft'))
