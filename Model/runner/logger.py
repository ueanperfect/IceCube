import os
import datetime
import torch
import logging


class Logger():
    def __init__(self, path, model_name):
        path_dirs = os.listdir(os.getcwd())
        if path not in path_dirs:
            os.mkdir(path)
        self.path = path
        self.model_name = model_name
        self.current_path = self.check_or_make_dir()
        self.set_logging_configer()

    def check_or_make_dir(self):
        logs_list = os.listdir(self.path)
        if self.model_name not in logs_list:
            os.mkdir(os.path.join(self.path, self.model_name))
        data_time = str(datetime.datetime.now())
        work_dir = os.path.join(self.path, self.model_name, data_time)
        os.mkdir(work_dir)
        os.mkdir(os.path.join(work_dir, 'checkpoints'))
        os.mkdir(os.path.join(work_dir, 'logger'))
        return work_dir

    def set_logging_configer(self):
        logger_file_name = self.model_name + '.log'
        logging_path = os.path.join(self.current_path, 'logger', logger_file_name)
        logging.basicConfig(filename=logging_path, format='%(asctime)s-%(levelname)s: %(message)s', level=logging.DEBUG)

    def save_checkpoint(self, model, epoch_number):
        checkpoints_name = "{}_{}_{}".format(model.model_name, str(epoch_number+1), '.pth')
        torch.save(model.state_dict(), os.path.join(self.current_path, 'checkpoints', checkpoints_name))
        logging.info("{}_{} has been saved \n".format(model.model_name, epoch_number))
        print("{}_{} has been saved \n".format(model.model_name, epoch_number))

    def print_evaluate_information(self, loss, acc):
        logging.info(f"Test Error: \n Accuracy: {(100 * acc):>0.1f}%, Avg loss: {loss:>8f}")
        print(f"Test Error: \n Accuracy: {(100 * acc):>0.1f}%, Avg loss: {loss:>8f}")

    def print_training_information(self, loss, current, size):
        logging.info(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def show_progress(self,epoch):
        logging.info(f"Start training the {epoch}st epoch")
        print(f"Start training the {epoch}st epoch")