import torch
from tqdm import tqdm
from src.training_utils import compute_loss, validate, move_data_to_device, MetricsStorage
from src.pipeline_builder import PipelineBuilder
from src.model_io import save_checkpoint

def train(builder:PipelineBuilder):
        cls_weight = builder.config.training.cls_weight    
        num_epochs = builder.config.training.num_epochs
        needs_validate = builder.config.training.validate
        val_interval = builder.config.training.val_interval
        train_dataloader, val_dataloader, _ = builder.build_dataloader()
        if builder.config.training.log:
            builder.training_summary(len(train_dataloader.dataset))
        device = builder.device
        model = builder.build_model()
        if builder.config.training.log:
            print(model.param_summary())
        if builder.config.model.compilate:
            model = torch.compile(model) 
        optimizer = builder.build_optimizer(model)
        scheduler = builder.build_scheduler(optimizer, train_dataloader)
        first_time = True
        metrics_storage = MetricsStorage(filepath=builder.config.training.metric_file, 
                                         decisive_metric=builder.config.training.decisive_metric)
        for epoch in range(num_epochs):
            model.train()  # Set the model to training mode
            metrics_storage.init_epoch()
            if first_time and builder.config.training.log:
                print('starting data loading')
            for batch in tqdm(train_dataloader):#
                # LOAD DATA TO DEVICE
                x,x_mask,y, y_mask, fixation_len = move_data_to_device(batch, device)
                optimizer.zero_grad()  # Zero the gradients
                if first_time and builder.config.training.log and builder.config.model.compilate:
                    print('model compilation')
                    first_time = False
                # FORWARD PASS AND LOSS COMPUTATION
                reg_out, cls_out = model(x,y, src_mask = x_mask, tgt_mask = y_mask)  # Forward pass
                cls_loss, reg_loss = compute_loss(reg_out,cls_out, y, y_mask, fixation_len) # Compute loss
                total_loss = (1-cls_weight)*reg_loss + cls_weight*cls_loss
                # BACKWARD PASS AND OPTIMIZATION
                total_loss.backward()
                optimizer.step()
                if builder.config.scheduler.batch_lr:
                    scheduler.step()
                metrics_storage.update_batch_loss(reg_loss, cls_loss)
            avg_reg_loss, avg_cls_loss = metrics_storage.finalize_epoch()
            print(f"Epoch {epoch+1}/{num_epochs}, Reg Loss: {avg_reg_loss:.4f}, Cls Loss: {avg_cls_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
            if not builder.config.scheduler.batch_lr:
                scheduler.step()
            if needs_validate and ((epoch + 1) % val_interval == 0):
                validate(model, val_dataloader, epoch, device, metrics_storage.metrics, log = builder.config.training.log)
                metrics_storage.save_metrics()
                is_best = metrics_storage.update_best()
                if is_best:
                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        filepath=builder.config.training.checkpoint_file,
                        epoch=epoch,
                        log=builder.config.training.log,
                        save_full_state= builder.config.training.save_full_state
                    )

        print("Training finished!")