import torch
import wandb
import os
from datetime import datetime
import lightning as L
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
import logging
import argparse

# Import your data modules
from lightning_modules.dataset_module import DatasetModule
# Import your model
from  lightning_modules.model_module import MyModel


def main(args):

    DEBUG_MODE = args.debug
    batch_size = args.batch_size
    max_epochs = args.epochs
    data_name = args.dataset
    image_size = args.image_size
    model_name = args.model_name
    model_size = args.model_size
    lr = args.lr
    mapping_num_classes={
        "CRDDC":2,
        "SVRDD":7,
        "RTK":3,
        "KJ":5,
        "Road_CLS_Quality":7

    }
    num_classes = mapping_num_classes[data_name]
    run_name = f"{model_name}_{model_size}_{data_name}"
    project_name = "ResEViT_Road"
    log_directory = "./wandb_logs"



    timestamp = datetime.now().strftime("%m-%d--%H-%M")
    seed_everything(42, workers=True)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    if DEBUG_MODE:
        logger.setLevel(logging.DEBUG)
        os.environ["WANDB_SILENT"] = "true"
        os.environ["WANDB_MODE"] = "disabled"
        os.environ["WANDB_LOCAL_LOG"] = "off"
        dev_run = True
    else:
        os.environ["WANDB_SILENT"] = "false"
        os.environ["WANDB_MODE"] = "online"
        dev_run = False

    data_module = DatasetModule(dataset_name=data_name, batch_size=batch_size, image_size=image_size)
    # logger.info("Train length: %s", len(data_module.train_dataloader()) * batch_size)
    # logger.info("Valid length: %s", len(data_module.val_dataloader()) * batch_size)

    # Initialize Callbacks
    early_stop_callback = EarlyStopping(monitor="val_loss", verbose=True, patience=15)
    checkpoint_callback = ModelCheckpoint(filename='{epoch:02d}-{val_acc:.4f}', monitor='val_acc',
                                          mode='max', save_top_k=5, save_last=True)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    wandb_logger = WandbLogger(name=run_name, save_dir=log_directory, project=project_name, log_model=False,
                               version=timestamp, job_type='train', mode=("disabled" if DEBUG_MODE else "online"))

    # Initialize Trainer
    trainer = L.Trainer(max_epochs=max_epochs,
                        logger=wandb_logger,
                        fast_dev_run=dev_run,
                        accelerator='gpu',
                        callbacks=[lr_monitor,
                                   early_stop_callback,
                                   checkpoint_callback]
                        )
    train_model = MyModel(model_name=model_name, model_size=model_size, num_classes=num_classes, lr=lr,
                          image_size=image_size)
    Model=MyModel

    # Fit model
    data_module.setup("fit")
    trainer.fit(train_model, datamodule=data_module)

    data_module.setup("test")
    # logger.info("Test length: %s", len(data_module.test_dataloader()) * batch_size)

    best_model_path = trainer.checkpoint_callback.best_model_path
    best_model = Model.load_from_checkpoint(best_model_path)
    test_result = trainer.test(best_model, dataloaders=data_module.test_dataloader())
    logger.info("Test_result best_model_path: %s", test_result)
    logger.info('best_model_path: %s', best_model_path)

    os.makedirs(f"./models/{data_name}", exist_ok=True)
    torch.save(best_model, f"./models/{data_name}/{run_name}-{timestamp}.pt")
    torch.save(best_model.state_dict(), f"./models/{data_name}/{run_name}-{timestamp}-state_dict.pt")

    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model in ResEViT-road project')
    parser.add_argument('--dataset', type=str, choices=['CRDDC','SVRDD',"RTK","KJ","Road_CLS_Quality"], required=True, help='Dataset to use CRDDC')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--image_size', type=int, default=224, help='Image size for training')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--model_name', type=str, default="resevit", help='Model name to use')
    parser.add_argument('--model_size', type=str, default="small", help='Model size to use')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for training')
    args = parser.parse_args()

    main(args)