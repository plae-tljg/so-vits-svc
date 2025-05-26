import logging
import multiprocessing
import os
import time

import torch
import torch_musa
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import modules.commons as commons
import utils
from data_utils import TextAudioCollate, TextAudioSpeakerLoader
from models import (
    MultiPeriodDiscriminator,
    SynthesizerTrn,
)
from modules.losses import discriminator_loss, feature_loss, generator_loss, kl_loss
from modules.mel_processing import mel_spectrogram_torch, spec_to_mel_torch

logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('numba').setLevel(logging.WARNING)

torch.backends.cudnn.benchmark = True
global_step = 0
start_time = time.time()

def main():
    """Single GPU Training with MUSA"""
    hps = utils.get_hparams()
    device = torch.device("musa")
    
    logger = utils.get_logger(hps.model_dir)
    logger.info(hps)
    utils.check_git_hash(hps.model_dir)
    writer = SummaryWriter(log_dir=hps.model_dir)
    writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))
    
    torch.manual_seed(hps.train.seed)
    collate_fn = TextAudioCollate()
    all_in_mem = hps.train.all_in_mem
    train_dataset = TextAudioSpeakerLoader(hps.data.training_files, hps, all_in_mem=all_in_mem)
    num_workers = 5 if multiprocessing.cpu_count() > 4 else multiprocessing.cpu_count()
    if all_in_mem:
        num_workers = 0
    train_loader = DataLoader(train_dataset, num_workers=num_workers, shuffle=False, pin_memory=True,
                              batch_size=hps.train.batch_size, collate_fn=collate_fn)
    
    eval_dataset = TextAudioSpeakerLoader(hps.data.validation_files, hps, all_in_mem=all_in_mem, vol_aug=False)
    eval_loader = DataLoader(eval_dataset, num_workers=1, shuffle=False,
                             batch_size=1, pin_memory=False,
                             drop_last=False, collate_fn=collate_fn)

    net_g = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).to(device)
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).to(device)
    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps)
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps)

    skip_optimizer = False
    try:
        _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g,
                                                   optim_g, skip_optimizer)
        _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d,
                                                   optim_d, skip_optimizer)
        epoch_str = max(epoch_str, 1)
        name = utils.latest_checkpoint_path(hps.model_dir, "D_*.pth")
        global_step = int(name[name.rfind("_")+1:name.rfind(".")])+1
    except Exception:
        print("load old checkpoint failed...")
        epoch_str = 1
        global_step = 0
    if skip_optimizer:
        epoch_str = 1
        global_step = 0

    warmup_epoch = hps.train.warmup_epochs
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)

    scaler = GradScaler(enabled=hps.train.fp16_run)

    for epoch in range(epoch_str, hps.train.epochs + 1):
        if epoch <= warmup_epoch:
            for param_group in optim_g.param_groups:
                param_group['lr'] = hps.train.learning_rate / warmup_epoch * epoch
            for param_group in optim_d.param_groups:
                param_group['lr'] = hps.train.learning_rate / warmup_epoch * epoch
        
        train_and_evaluate(device, epoch, hps, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d], scaler,
                           [train_loader, eval_loader], logger, [writer, writer_eval])
        
        scheduler_g.step()
        scheduler_d.step()


def train_and_evaluate(device, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers):
    net_g, net_d = nets
    optim_g, optim_d = optims
    scheduler_g, scheduler_d = schedulers
    train_loader, eval_loader = loaders
    if writers is not None:
        writer, writer_eval = writers
    
    half_type = torch.bfloat16 if hps.train.half_type=="bf16" else torch.float16
    global global_step

    # 添加错误计数器和设备状态跟踪
    error_count = 0
    max_errors_per_epoch = 100
    consecutive_errors = 0
    max_consecutive_errors = 20
    device_status = "musa"  # 跟踪当前设备状态
    current_device = device  # 保存初始设备
    gpu_error_count = 0  # 跟踪GPU错误次数
    max_gpu_errors_before_fallback = 50  # GPU错误阈值

    # 确保模型在GPU上
    try:
        net_g.to(current_device)
        net_d.to(current_device)
        print("Models successfully loaded to MUSA device")
    except RuntimeError as e:
        print(f"Warning: Failed to load models to MUSA: {str(e)}")
        current_device = torch.device("cpu")
        device_status = "cpu"
        net_g.to(current_device)
        net_d.to(current_device)

    net_g.train()
    net_d.train()

    def reset_device_status():
        """重置所有模型到指定设备"""
        nonlocal device_status, current_device, gpu_error_count
        if device_status == "cpu" and gpu_error_count < max_gpu_errors_before_fallback:
            try:
                print("Attempting to move models back to MUSA...")
                net_g.to(device)
                net_d.to(device)
                current_device = device
                device_status = "musa"
                gpu_error_count = 0  # 重置GPU错误计数
                print("Successfully reset models to MUSA device")
            except RuntimeError as e:
                print(f"Warning: Failed to reset to MUSA: {str(e)}")
                gpu_error_count += 1
                if gpu_error_count >= max_gpu_errors_before_fallback:
                    print("Too many GPU errors, staying on CPU")
                    current_device = torch.device("cpu")
                    net_g.to(current_device)
                    net_d.to(current_device)
                    device_status = "cpu"

    def safe_to_device(tensor, target_device):
        """安全地将张量转移到目标设备"""
        nonlocal device_status, current_device, gpu_error_count
        if tensor.device == target_device:
            return tensor
            
        try:
            # 优先尝试GPU
            if target_device.type == "musa" and gpu_error_count < max_gpu_errors_before_fallback:
                try:
                    return tensor.to(target_device, non_blocking=True)  # 启用异步传输
                except RuntimeError as e:
                    if "MUSA error" in str(e):
                        print(f"Warning: MUSA transfer failed: {str(e)}")
                        gpu_error_count += 1
                        if gpu_error_count >= max_gpu_errors_before_fallback:
                            print("Too many GPU errors, switching to CPU")
                            device_status = "cpu"
                            current_device = torch.device("cpu")
                        return tensor.cpu()
                    raise e
            # CPU回退
            else:
                return tensor.cpu()
        except Exception as e:
            print(f"Warning: Error in tensor transfer: {str(e)}")
            if "MUSA error" in str(e):
                gpu_error_count += 1
                if gpu_error_count >= max_gpu_errors_before_fallback:
                    device_status = "cpu"
                    current_device = torch.device("cpu")
                return tensor.cpu()
            raise e

    def ensure_same_device(tensors):
        """确保所有张量在同一设备上"""
        if not tensors:
            return tensors
        # 优先使用GPU
        if device_status == "musa" and gpu_error_count < max_gpu_errors_before_fallback:
            target_device = device
        else:
            target_device = torch.device("cpu")
        return [safe_to_device(t, target_device) for t in tensors]

    # 设置数据加载器
    train_loader = DataLoader(
        train_loader.dataset,
        batch_size=hps.train.batch_size,
        shuffle=True,
        num_workers=4,  # 增加工作进程数
        pin_memory=True,  # 启用内存固定
        persistent_workers=True  # 保持工作进程存活
    )

    for batch_idx, items in enumerate(train_loader):
        try:
            # 每100个batch尝试重置到GPU
            if batch_idx % 100 == 0:
                reset_device_status()
                
            c, f0, spec, y, spk, lengths, uv, volume = items
            
            # 批量转移到设备
            tensors_to_transfer = [c, f0, spec, y, spk, lengths, uv]
            if volume is not None:
                tensors_to_transfer.append(volume)
                
            # 使用异步传输
            with torch.cuda.stream(torch.cuda.Stream()):
                c, f0, spec, y, spk, lengths, uv, *volume_list = ensure_same_device(tensors_to_transfer)
                volume = volume_list[0] if volume_list else None
                
                # 确保所有张量在同一设备上
                current_device = c.device
                g = safe_to_device(spk, current_device)
            
            # 数值检查函数
            def check_and_fix_tensor(tensor, name, min_val=-1e4, max_val=1e4):
                if tensor is None:
                    return None
                if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                    print(f"Warning: {name} contains NaN/Inf, fixing...")
                    tensor = torch.nan_to_num(tensor, nan=0.0, posinf=max_val, neginf=min_val)
                    tensor = torch.clamp(tensor, min=min_val, max=max_val)
                return tensor

            # 检查并修复所有输入
            c = check_and_fix_tensor(c, "c")
            f0 = check_and_fix_tensor(f0, "f0")
            spec = check_and_fix_tensor(spec, "spec")
            y = check_and_fix_tensor(y, "y")
            uv = check_and_fix_tensor(uv, "uv")
            if volume is not None:
                volume = check_and_fix_tensor(volume, "volume")

            try:
                # 使用异步计算
                with torch.cuda.stream(torch.cuda.Stream()):
                    mel = spec_to_mel_torch(
                        spec,
                        hps.data.filter_length,
                        hps.data.n_mel_channels,
                        hps.data.sampling_rate,
                        hps.data.mel_fmin,
                        hps.data.mel_fmax)
                
                with autocast(enabled=hps.train.fp16_run, dtype=half_type):
                    # 确保模型在当前设备上
                    if next(net_g.parameters()).device != current_device:
                        net_g.to(current_device)
                    if next(net_d.parameters()).device != current_device:
                        net_d.to(current_device)
                        
                    # 使用异步计算
                    with torch.cuda.stream(torch.cuda.Stream()):
                        y_hat, ids_slice, z_mask, \
                        (z, z_p, m_p, logs_p, m_q, logs_q), pred_lf0, norm_lf0, lf0 = net_g(c, f0, uv, spec, g=g, c_lengths=lengths,
                                                                                            spec_lengths=lengths, vol=volume)

                        # 检查生成器输出
                        y_hat = check_and_fix_tensor(y_hat, "y_hat")

                        y_mel = commons.slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
                        y_hat_mel = mel_spectrogram_torch(
                            y_hat.squeeze(1),
                            hps.data.filter_length,
                            hps.data.n_mel_channels,
                            hps.data.sampling_rate,
                            hps.data.hop_length,
                            hps.data.win_length,
                            hps.data.mel_fmin,
                            hps.data.mel_fmax
                        )
                        y = commons.slice_segments(y, ids_slice * hps.data.hop_length, hps.train.segment_size)

                        # 确保所有张量在同一设备上
                        y_mel, y_hat_mel, y, y_hat = ensure_same_device([y_mel, y_hat_mel, y, y_hat])

                        # Discriminator
                        y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())

                        with autocast(enabled=False, dtype=half_type):
                            # 判别器损失
                            loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
                            loss_disc_all = loss_disc * 0.25
                            
                            # 检查判别器损失
                            if torch.isnan(loss_disc_all) or torch.isinf(loss_disc_all):
                                print(f"Warning: Discriminator loss is NaN/Inf, using fallback value")
                                loss_disc_all = torch.tensor(0.1, device=current_device)
            
            except RuntimeError as e:
                if "MUSA error" in str(e) or "Index should be on GPU device" in str(e):
                    print(f"Warning: Device error in forward pass: {str(e)}")
                    gpu_error_count += 1
                    if gpu_error_count >= max_gpu_errors_before_fallback:
                        print("Too many GPU errors, switching to CPU")
                        device_status = "cpu"
                        current_device = torch.device("cpu")
                        net_g.to(current_device)
                        net_d.to(current_device)
                    error_count += 1
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        print("Too many consecutive errors, saving checkpoint and resetting...")
                        try:
                            # 保存前确保模型在CPU上
                            net_g_cpu = net_g.cpu()
                            net_d_cpu = net_d.cpu()
                            utils.save_checkpoint(net_g_cpu, optim_g, hps.train.learning_rate, epoch,
                                                os.path.join(hps.model_dir, f"G_{global_step}_error.pth"))
                            utils.save_checkpoint(net_d_cpu, optim_d, hps.train.learning_rate, epoch,
                                                os.path.join(hps.model_dir, f"D_{global_step}_error.pth"))
                            # 重置设备状态
                            reset_device_status()
                            consecutive_errors = 0
                        except Exception as save_error:
                            print(f"Warning: Failed to save checkpoint: {str(save_error)}")
                    continue
                raise e

            # 判别器更新
            try:
                optim_d.zero_grad()
                scaler.scale(loss_disc_all).backward()
                
                # 梯度裁剪
                scaler.unscale_(optim_d)
                grad_norm_d = commons.clip_grad_value_(net_d.parameters(), 0.5)
                scaler.step(optim_d)
                
                with autocast(enabled=hps.train.fp16_run, dtype=half_type):
                    # Generator
                    y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
                    with autocast(enabled=False, dtype=half_type):
                        # 生成器损失
                        loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                        loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
                        loss_fm = feature_loss(fmap_r, fmap_g) * 0.25
                        loss_gen, losses_gen = generator_loss(y_d_hat_g)
                        loss_lf0 = F.mse_loss(pred_lf0, lf0) if net_g.use_automatic_f0_prediction else 0
                        
                        # 损失裁剪
                        loss_mel = torch.clamp(loss_mel, max=50.0)
                        loss_kl = torch.clamp(loss_kl, max=5.0)
                        loss_fm = torch.clamp(loss_fm, max=5.0)
                        loss_gen = torch.clamp(loss_gen, max=50.0)
                        loss_lf0 = torch.clamp(loss_lf0, max=5.0)
                        
                        loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl + loss_lf0
                        
                        # 检查生成器损失
                        if torch.isnan(loss_gen_all) or torch.isinf(loss_gen_all):
                            print(f"Warning: Generator loss is NaN/Inf, using fallback value")
                            loss_gen_all = torch.tensor(0.1, device=current_device)
                
                # 生成器更新
                optim_g.zero_grad()
                scaler.scale(loss_gen_all).backward()
                
                # 梯度裁剪
                scaler.unscale_(optim_g)
                grad_norm_g = commons.clip_grad_value_(net_g.parameters(), 0.5)
                scaler.step(optim_g)
                scaler.update()

                # 重置连续错误计数
                consecutive_errors = 0

            except RuntimeError as e:
                if "MUSA error" in str(e) or "Index should be on GPU device" in str(e):
                    print(f"Warning: Device error in backward pass: {str(e)}")
                    gpu_error_count += 1
                    if gpu_error_count >= max_gpu_errors_before_fallback:
                        print("Too many GPU errors, switching to CPU")
                        device_status = "cpu"
                        current_device = torch.device("cpu")
                        net_g.to(current_device)
                        net_d.to(current_device)
                    error_count += 1
                    consecutive_errors += 1
                    # 清理梯度
                    optim_g.zero_grad()
                    optim_d.zero_grad()
                    continue
                raise e

            # 检查错误计数
            if error_count >= max_errors_per_epoch:
                print(f"Warning: Too many errors in this epoch ({error_count}), saving checkpoint and continuing...")
                try:
                    # 保存前确保模型在CPU上
                    net_g_cpu = net_g.cpu()
                    net_d_cpu = net_d.cpu()
                    utils.save_checkpoint(net_g_cpu, optim_g, hps.train.learning_rate, epoch,
                                        os.path.join(hps.model_dir, f"G_{global_step}_error.pth"))
                    utils.save_checkpoint(net_d_cpu, optim_d, hps.train.learning_rate, epoch,
                                        os.path.join(hps.model_dir, f"D_{global_step}_error.pth"))
                    # 重置设备状态
                    reset_device_status()
                    error_count = 0
                except Exception as save_error:
                    print(f"Warning: Failed to save checkpoint: {str(save_error)}")
                continue

            if global_step % hps.train.log_interval == 0:
                lr = optim_g.param_groups[0]['lr']
                losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_kl]
                reference_loss = sum(losses)
                logger.info('Train Epoch: {} [{:.0f}%]'.format(
                    epoch,
                    100. * batch_idx / len(train_loader)))
                logger.info(f"Losses: {[x.item() for x in losses]}, step: {global_step}, lr: {lr}, reference_loss: {reference_loss}")

                scalar_dict = {"loss/g/total": loss_gen_all, "loss/d/total": loss_disc_all, "learning_rate": lr,
                               "grad_norm_d": grad_norm_d, "grad_norm_g": grad_norm_g}
                scalar_dict.update({"loss/g/fm": loss_fm, "loss/g/mel": loss_mel, "loss/g/kl": loss_kl,
                                    "loss/g/lf0": loss_lf0})

                image_dict = {
                    "slice/mel_org": utils.plot_spectrogram_to_numpy(y_mel[0].data.cpu().numpy()),
                    "slice/mel_gen": utils.plot_spectrogram_to_numpy(y_hat_mel[0].data.cpu().numpy()),
                    "all/mel": utils.plot_spectrogram_to_numpy(mel[0].data.cpu().numpy())
                }

                if net_g.use_automatic_f0_prediction:
                    image_dict.update({
                        "all/lf0": utils.plot_data_to_numpy(lf0[0, 0, :].cpu().numpy()),
                        "all/norm_lf0": utils.plot_data_to_numpy(lf0[0, 0, :].cpu().numpy())
                    })

                utils.summarize(
                    writer=writer,
                    global_step=global_step,
                    images=image_dict,
                    scalars=scalar_dict
                )

            if global_step % hps.train.eval_interval == 0:
                evaluate(hps, net_g, eval_loader, writer_eval)
                utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch,
                                      os.path.join(hps.model_dir, "G_{}.pth".format(global_step)))
                utils.save_checkpoint(net_d, optim_d, hps.train.learning_rate, epoch,
                                      os.path.join(hps.model_dir, "D_{}.pth".format(global_step)))
                keep_ckpts = getattr(hps.train, 'keep_ckpts', 0)
                if keep_ckpts > 0:
                    utils.clean_checkpoints(path_to_models=hps.model_dir, n_ckpts_to_keep=keep_ckpts, sort_by_time=True)

            global_step += 1

        except Exception as e:
            print(f"Warning: Unexpected error in training loop: {str(e)}")
            error_count += 1
            consecutive_errors += 1
            if consecutive_errors >= max_consecutive_errors:
                print("Too many consecutive errors, saving checkpoint and resetting...")
                try:
                    # 保存前确保模型在CPU上
                    net_g_cpu = net_g.cpu()
                    net_d_cpu = net_d.cpu()
                    utils.save_checkpoint(net_g_cpu, optim_g, hps.train.learning_rate, epoch,
                                        os.path.join(hps.model_dir, f"G_{global_step}_error.pth"))
                    utils.save_checkpoint(net_d_cpu, optim_d, hps.train.learning_rate, epoch,
                                        os.path.join(hps.model_dir, f"D_{global_step}_error.pth"))
                    # 重置设备状态
                    reset_device_status()
                    consecutive_errors = 0
                except Exception as save_error:
                    print(f"Warning: Failed to save checkpoint: {str(save_error)}")
            continue

    if global_step % hps.train.eval_interval == 0:
        evaluate(hps, net_g, eval_loader, writer_eval)
        utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch,
                              os.path.join(hps.model_dir, "G_{}.pth".format(global_step)))
        utils.save_checkpoint(net_d, optim_d, hps.train.learning_rate, epoch,
                              os.path.join(hps.model_dir, "D_{}.pth".format(global_step)))
        keep_ckpts = getattr(hps.train, 'keep_ckpts', 0)
        if keep_ckpts > 0:
            utils.clean_checkpoints(path_to_models=hps.model_dir, n_ckpts_to_keep=keep_ckpts, sort_by_time=True)

    if global_step % hps.train.log_interval == 0:
        global start_time
        now = time.time()
        durtaion = format(now - start_time, '.2f')
        logger.info(f'====> Epoch: {epoch}, cost {durtaion} s')
        start_time = now


def evaluate(hps, generator, eval_loader, writer_eval):
    generator.eval()
    device = next(generator.parameters()).device
    image_dict = {}
    audio_dict = {}
    with torch.no_grad():
        for batch_idx, items in enumerate(eval_loader):
            c, f0, spec, y, spk, _, uv, volume = items
            g = spk[:1].to(device)
            spec, y = spec[:1].to(device), y[:1].to(device)
            c = c[:1].to(device)
            f0 = f0[:1].to(device)
            uv = uv[:1].to(device)
            if volume is not None:
                volume = volume[:1].to(device)
            mel = spec_to_mel_torch(
                spec,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.mel_fmin,
                hps.data.mel_fmax)
            y_hat, _ = generator.infer(c, f0, uv, g=g, vol=volume)

            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1).float(),
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax
            )

            audio_dict.update({
                f"gen/audio_{batch_idx}": y_hat[0],
                f"gt/audio_{batch_idx}": y[0]
            })
        image_dict.update({
            "gen/mel": utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy()),
            "gt/mel": utils.plot_spectrogram_to_numpy(mel[0].cpu().numpy())
        })
    utils.summarize(
        writer=writer_eval,
        global_step=global_step,
        images=image_dict,
        audios=audio_dict,
        audio_sampling_rate=hps.data.sampling_rate
    )
    generator.train()


if __name__ == "__main__":
    main()