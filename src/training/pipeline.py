import torch
from tqdm import tqdm
from src.training.training_utils import compute_loss, validate, move_data_to_device, MetricsStorage
from src.training.pipeline_builder import PipelineBuilder
from src.model.model_io import save_checkpoint, save_splits

def train(builder:PipelineBuilder):
        phases = builder.build_phases()
        needs_validate = builder.config.training.validate
        val_interval = builder.config.training.val_interval
        builder.load_dataset()
        device = builder.device
        model, splits = builder.build_model()
        rec_interval = builder.config.training.rec_interval
        inference_recorder = builder.build_inference_recorder(model)
        if splits is not None:
            print("Loading splits from pretrained model")
            train_idx, val_idx, test_idx = splits
        else:
            train_idx, val_idx, test_idx = builder.make_splits()
        save_splits(train_idx, val_idx, test_idx, builder.config.training.splits_file)
        if builder.config.training.log:
            print(f"Split saved to {builder.config.training.splits_file}")
            train_dataloader, val_dataloader, _ = builder.build_dataloader(train_idx, val_idx, test_idx)
        builder.clear_dataframe()

        if builder.config.training.log:
            builder.training_summary(len(train_dataloader.dataset))

        if builder.config.training.log:
            print(model.param_summary())
        
        if builder.config.model.compilate and inference_recorder is None:
            model = torch.compile(model, dynamic=True) 
        elif builder.config.model.compilate and inference_recorder is not None:
            print("Inference recorder enabled: skipping torch.compile to preserve module-level captures.")
        to_update_in_epoch = []
        to_update_in_batch = []
        curriculum_noise = None
        if builder.curriculum_noise is not None:
            curriculum_noise = builder.curriculum_noise
            builder.curriculum_noise.set_steps_per_epoch(len(train_dataloader))
            to_update_in_batch.append(builder.curriculum_noise)
        optimizer = builder.build_optimizer(model)
        scheduler = builder.build_scheduler(optimizer, train_dataloader)
        if builder.config.scheduler.batch_lr:
            to_update_in_batch.append(scheduler)
        else:
            to_update_in_epoch.append(scheduler)
        loss_fn = builder.build_loss_fn()
        first_time = True
        metrics_storage = MetricsStorage(filepath=builder.config.training.metric_file, 
                                         decisive_metric=builder.config.training.decisive_metric)
        
        weights_scheduler = builder.build_weights_scheduler(loss_fn)
        if weights_scheduler is not None:
            to_update_in_epoch.append(weights_scheduler)
        
        scheduled_sampling = builder.build_scheduled_sampling(len(train_dataloader))
        if scheduled_sampling is not None:
            to_update_in_batch.append(scheduled_sampling)
            model.set_scheduled_sampling(scheduled_sampling)
        denoise_dropout_scheduler = builder.build_denoise_dropout_scheduler(model, len(train_dataloader))
        if denoise_dropout_scheduler is not None:
            to_update_in_batch.append(denoise_dropout_scheduler)
        global_epoch = 0
        global_step = 0
        record_train_split = (
            inference_recorder is not None
            and builder.config.training.inference_recorder.get('split', 'train') in ('train', 'both')
        )
        record_val_split = (
            inference_recorder is not None
            and builder.config.training.inference_recorder.get('split', 'train') in ('val', 'both')
        )

        for phase, denoise_weight, decisive_metric, epochs in phases:
            print(f"Training {phase} for {epochs} epochs, Denoise Weight: {denoise_weight}")
            model.set_phase(phase)
            loss_fn.set_denoise_weight(denoise_weight)
            metrics_storage.decisive_metric = decisive_metric
            loss_fn.summary()
            for epoch in range(epochs):
                model.train()  # Set the model to training mode
                metrics_storage.init_epoch()
                if first_time and builder.config.training.log:
                    print('starting data loading')

                if epoch > 0 and epoch%rec_interval == 0:
                    record_index = len(train_dataloader)-1
                else:
                    record_index = -1
                for batch_index, batch in enumerate(tqdm(train_dataloader)):#
                    # LOAD DATA TO DEVICE
                    input = move_data_to_device(batch, device)
                    if record_train_split and record_index == batch_index:
                        inference_recorder.enabled = True
                        inference_recorder.start_batch(
                            epoch=global_epoch + 1,
                            phase=phase,
                            split='train',
                            batch_index=batch_index,
                            global_step=global_step,
                            metadata={'model_name': model.name, 'phase_epoch': epoch + 1},
                        )
                    else:
                        inference_recorder.enabled = False
                    optimizer.zero_grad()  # Zero the gradients
                    if first_time and builder.config.training.log and builder.config.model.compilate and inference_recorder is None:
                        print('model compilation')
                        first_time = False
                    # FORWARD PASS AND LOSS COMPUTATION
                    output = model(**input)  # Forward pass
                    if record_train_split:
                        inference_recorder.record_batch(input, output)
                        inference_recorder.save_batch()
                    loss, info = loss_fn(input, output) # Compute loss
                    # BACKWARD PASS AND OPTIMIZATION
                    loss.backward()
                    optimizer.step()
                    for update in to_update_in_batch:
                        if update is not None:
                            update.step()
                    metrics_storage.update_batch_loss(info)
                    metrics_storage.compute_normalized_regression_metrics(input, output, train_dataloader)
                    global_step += 1
                loss_info = metrics_storage.finalize_epoch()
                loss_str = ", ".join([f"{key}: {value:.4f}" for key, value in loss_info.items()])
                print(f"Epoch {epoch+1}/{epochs}, {loss_str}, LR: {optimizer.param_groups[0]['lr']:.6f} ",
                      f"Scheduled Sampling: {scheduled_sampling.get_current_ratio()}" if scheduled_sampling is not None else "",
                      f"Curriculum Noise: {curriculum_noise.get_alpha()}" if curriculum_noise is not None else "",
                      f"Denoise Dropout: {denoise_dropout_scheduler.get_prob()}" if denoise_dropout_scheduler is not None else "")
                for updater in to_update_in_epoch:
                    if updater is not None:
                        updater.step()
                if needs_validate and ((epoch + 1) % val_interval == 0):
                    if curriculum_noise is not None:
                        curriculum_noise.enabled = False
                    validate(
                        model,
                        loss_fn,
                        val_dataloader,
                        global_epoch,
                        device,
                        metrics_storage.metrics,
                        log = builder.config.training.log,
                        inference_recorder = inference_recorder if record_val_split and record_index > 0 else None,
                        recorder_phase = phase,
                    )
                    if curriculum_noise is not None:
                        curriculum_noise.enabled = True
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
                global_epoch += 1
            
        print("Training finished!")
