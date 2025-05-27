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

    # 修改错误处理参数
    error_count = 0
    max_errors_per_epoch = 50
    consecutive_errors = 0
    max_consecutive_errors = 10
    device_status = "cpu"  # 默认使用CPU
    current_device = torch.device("cpu")
    gpu_error_count = 0
    max_gpu_errors_before_fallback = 5
    recovery_attempts = 0
    max_recovery_attempts = 3

    def save_checkpoint_safe():
        """安全地保存检查点，确保在CPU上操作"""
        try:
            # 将模型移动到CPU
            net_g_cpu = net_g.cpu()
            net_d_cpu = net_d.cpu()
            
            # 保存检查点
            utils.save_checkpoint(net_g_cpu, optim_g, hps.train.learning_rate, epoch,
                                os.path.join(hps.model_dir, f"G_{global_step}_recovery.pth"))
            utils.save_checkpoint(net_d_cpu, optim_d, hps.train.learning_rate, epoch,
                                os.path.join(hps.model_dir, f"D_{global_step}_recovery.pth"))
            
            # 如果之前是GPU模式，尝试移回GPU
            if device_status == "musa" and torch_musa.is_available():
                try:
                    net_g.to(device)
                    net_d.to(device)
                except RuntimeError:
                    print("Warning: Failed to move models back to MUSA after saving")
            
            return True
        except Exception as e:
            print(f"Warning: Failed to save checkpoint: {str(e)}")
            return False

    def recover_from_error():
        """从错误中恢复"""
        nonlocal recovery_attempts, consecutive_errors, error_count, device_status, current_device
        
        print(f"Attempting recovery (attempt {recovery_attempts + 1}/{max_recovery_attempts})...")
        
        # 保存当前状态
        if save_checkpoint_safe():
            print("Successfully saved recovery checkpoint")
        
        # 清理GPU内存
        if torch_musa.is_available():
            try:
                torch_musa.empty_cache()
            except:
                pass
        
        # 重置模型到CPU
        try:
            net_g.to(torch.device("cpu"))
            net_d.to(torch.device("cpu"))
            current_device = torch.device("cpu")
            device_status = "cpu"
            
            # 重置优化器状态
            optim_g.zero_grad()
            optim_d.zero_grad()
            
            # 重置错误计数
            consecutive_errors = 0
            error_count = 0
            gpu_error_count = 0
            
            recovery_attempts += 1
            return True
        except Exception as e:
            print(f"Warning: Recovery failed: {str(e)}")
            return False

    # 初始化模型
    try:
        net_g.to(current_device)
        net_d.to(current_device)
        print("Models loaded to CPU first")
        
        if torch_musa.is_available():
            try:
                net_g.to(device)
                net_d.to(device)
                current_device = device
                device_status = "musa"
                print("Models successfully moved to MUSA device")
            except RuntimeError as e:
                print(f"Warning: Failed to move models to MUSA: {str(e)}")
                print("Staying on CPU")
    except Exception as e:
        print(f"Warning: Error during device initialization: {str(e)}")
        print("Staying on CPU")

    net_g.train()
    net_d.train()

    def reset_device_status():
        """重置设备状态，但更保守的策略"""
        nonlocal device_status, current_device, gpu_error_count
        if device_status == "cpu" and gpu_error_count < max_gpu_errors_before_fallback:
            try:
                if torch_musa.is_available():
                    print("Attempting to move models to MUSA...")
                    net_g.to(device)
                    net_d.to(device)
                    current_device = device
                    device_status = "musa"
                    gpu_error_count = 0
                    print("Successfully moved models to MUSA device")
            except RuntimeError as e:
                print(f"Warning: Failed to move to MUSA: {str(e)}")
                gpu_error_count += 1
                if gpu_error_count >= max_gpu_errors_before_fallback:
                    print("Too many GPU errors, staying on CPU")
                    current_device = torch.device("cpu")
                    net_g.to(current_device)
                    net_d.to(current_device)
                    device_status = "cpu"

    def safe_to_device(tensor, target_device):
        """更安全的设备转移策略"""
        nonlocal device_status, current_device, gpu_error_count
        if tensor.device == target_device:
            return tensor
            
        try:
            # 如果目标设备是MUSA且错误次数未超限
            if target_device.type == "musa" and gpu_error_count < max_gpu_errors_before_fallback:
                try:
                    return tensor.to(target_device, non_blocking=False)  # 禁用异步传输
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
            # 默认使用CPU
            return tensor.cpu()
        except Exception as e:
            print(f"Warning: Error in tensor transfer: {str(e)}")
            return tensor.cpu()

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

    for batch_idx, items in enumerate(train_loader):
        try:
            # 检查是否需要恢复
            if consecutive_errors >= max_consecutive_errors:
                if recovery_attempts >= max_recovery_attempts:
                    print("Too many recovery attempts, stopping training")
                    return
                if not recover_from_error():
                    print("Recovery failed, stopping training")
                    return
                continue

            # 每100个batch尝试重置到GPU
            if batch_idx % 100 == 0 and device_status == "cpu" and torch_musa.is_available():
                try:
                    net_g.to(device)
                    net_d.to(device)
                    current_device = device
                    device_status = "musa"
                    print("Successfully moved models back to MUSA")
                except RuntimeError:
                    print("Failed to move models back to MUSA, staying on CPU")

            # 处理数据
            c, f0, spec, y, spk, lengths, uv, volume = items
            
            # 批量转移到设备
            tensors_to_transfer = [c, f0, spec, y, spk, lengths, uv]
            if volume is not None:
                tensors_to_transfer.append(volume)
            
            try:
                c, f0, spec, y, spk, lengths, uv, *volume_list = ensure_same_device(tensors_to_transfer)
                volume = volume_list[0] if volume_list else None
            except RuntimeError as e:
                print(f"Warning: Error during tensor transfer: {str(e)}")
                consecutive_errors += 1
                continue

            # 数值检查
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

            # 训练步骤
            try:
                with autocast(enabled=hps.train.fp16_run, dtype=half_type):
                    # 确保模型在当前设备上
                    if next(net_g.parameters()).device != current_device:
                        net_g.to(current_device)
                    if next(net_d.parameters()).device != current_device:
                        net_d.to(current_device)
                    
                    # 前向传播
                    y_hat, ids_slice, z_mask, \
                    (z, z_p, m_p, logs_p, m_q, logs_q), pred_lf0, norm_lf0, lf0 = net_g(c, f0, uv, spec, g=spk, 
                                                                                        c_lengths=lengths,
                                                                                        spec_lengths=lengths, 
                                                                                        vol=volume)
                    
                    # 检查生成器输出
                    y_hat = check_and_fix_tensor(y_hat, "y_hat")
                    
                    # 计算损失
                    mel = spec_to_mel_torch(spec, hps.data.filter_length, hps.data.n_mel_channels,
                                          hps.data.sampling_rate, hps.data.mel_fmin, hps.data.mel_fmax)
                    
                    y_mel = commons.slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
                    y_hat_mel = mel_spectrogram_torch(y_hat.squeeze(1), hps.data.filter_length,
                                                    hps.data.n_mel_channels, hps.data.sampling_rate,
                                                    hps.data.hop_length, hps.data.win_length,
                                                    hps.data.mel_fmin, hps.data.mel_fmax)
                    
                    y = commons.slice_segments(y, ids_slice * hps.data.hop_length, hps.train.segment_size)
                    
                    # 确保所有张量在同一设备上
                    y_mel, y_hat_mel, y, y_hat = ensure_same_device([y_mel, y_hat_mel, y, y_hat])
                    
                    # 判别器前向传播
                    y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
                    
                    # 计算损失
                    with autocast(enabled=False, dtype=half_type):
                        loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
                        loss_disc_all = loss_disc * 0.25
                        
                        if torch.isnan(loss_disc_all) or torch.isinf(loss_disc_all):
                            print("Warning: Discriminator loss is NaN/Inf, using fallback value")
                            loss_disc_all = torch.tensor(0.1, device=current_device)
                
                # 判别器更新
                optim_d.zero_grad()
                scaler.scale(loss_disc_all).backward()
                scaler.unscale_(optim_d)
                grad_norm_d = commons.clip_grad_value_(net_d.parameters(), 0.5)
                scaler.step(optim_d)
                
                # 生成器更新
                with autocast(enabled=hps.train.fp16_run, dtype=half_type):
                    y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
                    with autocast(enabled=False, dtype=half_type):
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
                        
                        if torch.isnan(loss_gen_all) or torch.isinf(loss_gen_all):
                            print("Warning: Generator loss is NaN/Inf, using fallback value")
                            loss_gen_all = torch.tensor(0.1, device=current_device)
                
                optim_g.zero_grad()
                scaler.scale(loss_gen_all).backward()
                scaler.unscale_(optim_g)
                grad_norm_g = commons.clip_grad_value_(net_g.parameters(), 0.5)
                scaler.step(optim_g)
                scaler.update()
                
                # 重置错误计数
                consecutive_errors = 0
                error_count = 0
                
            except RuntimeError as e:
                if "MUSA error" in str(e) or "Index should be on GPU device" in str(e):
                    print(f"Warning: Device error in training step: {str(e)}")
                    gpu_error_count += 1
                    if gpu_error_count >= max_gpu_errors_before_fallback:
                        print("Too many GPU errors, switching to CPU")
                        device_status = "cpu"
                        current_device = torch.device("cpu")
                        net_g.to(current_device)
                        net_d.to(current_device)
                    error_count += 1
                    consecutive_errors += 1
                    continue
                raise e

            # 记录训练状态
            if global_step % hps.train.log_interval == 0:
                lr = optim_g.param_groups[0]['lr']
                losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_kl]
                reference_loss = sum(losses)
                logger.info('Train Epoch: {} [{:.0f}%]'.format(
                    epoch,
                    100. * batch_idx / len(train_loader)))
                logger.info(f"Losses: {[x.item() for x in losses]}, step: {global_step}, lr: {lr}, reference_loss: {reference_loss}")

                # 记录到tensorboard
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

            # 保存检查点
            if global_step % hps.train.eval_interval == 0:
                evaluate(hps, net_g, eval_loader, writer_eval)
                if save_checkpoint_safe():
                    print(f"Successfully saved checkpoint at step {global_step}")
                keep_ckpts = getattr(hps.train, 'keep_ckpts', 0)
                if keep_ckpts > 0:
                    utils.clean_checkpoints(path_to_models=hps.model_dir, n_ckpts_to_keep=keep_ckpts, sort_by_time=True)

            global_step += 1

        except Exception as e:
            print(f"Warning: Unexpected error in training loop: {str(e)}")
            error_count += 1
            consecutive_errors += 1
            if consecutive_errors >= max_consecutive_errors:
                if not recover_from_error():
                    print("Recovery failed, stopping training")
                    return
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