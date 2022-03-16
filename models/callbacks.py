import os, xlsxwriter
from pytorch_lightning.callbacks import Callback
import pytorch_lightning as pl

class MyPrintingCallback(Callback):

    def on_init_end(self, trainer):
        print('trainer is init now')

    def on_train_epoch_end(self, trainer, pl_module):
        print('do something when on_train_epoch_end')


class CustomCSVLogger(Callback):

    def __init__(self, out_dir, name):
        self.out_dir = out_dir
        self.name = name

        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

    def on_train_end(self, trainer, pl_module):

        train_scores = pl_module.train_scores
        valid_scores = pl_module.val_scores

        _file = os.path.join(self.out_dir, self.name)
        workbook = xlsxwriter.Workbook(_file)

        train_worksheet = workbook.add_worksheet("train")
        for row, item in enumerate(train_scores):

            # first line will always be header
            if row == 0:
                headers = item.keys()
                for col, value in enumerate(headers):
                    train_worksheet.write(row + 1, col + 1, value)

            # start insert vlaues from 2nd row
            values = item.values()
            for col, value in enumerate(values):
                train_worksheet.write(row + 2, col + 1, value)

        valid_worksheet = workbook.add_worksheet("valid")
        for row, item in enumerate(valid_scores):

            # first line will always be header
            if row == 0:
                headers = item.keys()
                for col, value in enumerate(headers):
                    valid_worksheet.write(row + 1, col + 1, value)

            # start insert vlaues from 2nd row
            values = item.values()
            for col, value in enumerate(values):
                valid_worksheet.write(row + 2, col + 1, value)

        workbook.close()
        print(f"Writing train & valid scores to .xlsx \n Save Path: {_file}\n")


class CheckpointEveryNSteps(pl.Callback):
    """
    Save a checkpoint every N steps, instead of Lightning's default that checkpoints
    based on validation loss.
    """

    def __init__(
        self,
        save_step_frequency,
        prefix="N-Step-Checkpoint",
        use_modelcheckpoint_filename=False,
    ):
        """
        Args:
            save_step_frequency: how often to save in steps
            prefix: add a prefix to the name, only used if
                use_modelcheckpoint_filename=False
            use_modelcheckpoint_filename: just use the ModelCheckpoint callback's
                default filename, don't use ours.
        """
        self.save_step_frequency = save_step_frequency
        self.prefix = prefix
        self.use_modelcheckpoint_filename = use_modelcheckpoint_filename

    def on_batch_end(self, trainer: pl.Trainer, _):
        """ Check if we should save a checkpoint after every train batch """
        epoch = trainer.current_epoch
        global_step = trainer.global_step
        if global_step % self.save_step_frequency == 0:
            if self.use_modelcheckpoint_filename:
                filename = trainer.checkpoint_callback.filename
            else:
                filename = f"{self.prefix}_{epoch}_{global_step}.ckpt"
            ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, filename)
            trainer.save_checkpoint(ckpt_path)
