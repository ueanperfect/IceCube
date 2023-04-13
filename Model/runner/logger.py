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
        now = datetime.datetime.now()
        date_string = now.strftime("%Y-%m-%d_%H-%M-%S")
        work_dir = os.path.join(self.path, self.model_name, date_string)
        os.mkdir(work_dir)
        os.mkdir(os.path.join(work_dir, 'checkpoints'))
        os.mkdir(os.path.join(work_dir, 'logger'))
        return work_dir

    def set_logging_configer(self):
        logger_file_name = self.model_name + '.log'
        logging_path = os.path.join(self.current_path, 'logger', logger_file_name)
        logging.basicConfig(filename=logging_path, format='%(asctime)s-%(levelname)s: %(message)s', level=logging.DEBUG)

    def save_checkpoint(self, model,file_batch_index, epoch_number):
        checkpoints_name = "{}_{}_{}_{}".format(self.model_name,str(file_batch_index+1), str(epoch_number + 1), '.pth')
        torch.save(model.state_dict(), os.path.join(self.current_path, 'checkpoints', checkpoints_name))
        logging.info("{} has been saved \n".format(checkpoints_name))
        print("{} has been saved \n".format(checkpoints_name))

    def print_evaluate_information(self, loss, metric):
        logging.info(f"Val Result:\nsocore : {(metric):>8f}, val loss: {loss:>8f}")
        print(f"Val Result:\nsocore: {(metric):>8f}, val loss: {loss:>8f}")

    def print_training_information(self, loss, current, size):
        logging.info(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def show_file_progress(self, file_id: int, file_number: int):
        logging.info(f"Start training the {file_id}/{file_number} st file")
        print(f"Start training the {file_id}/{file_number} st file")

    def show_progress(self, epoch):
        logging.info(f"Start training the {epoch+1}st epoch")
        print(f"Start training the {epoch+1}st epoch")

    def running_batch_information(self,file_batch_id):
        logging.info(f"Start training the {file_batch_id+1}st file batch")
        print(f"Start training the {file_batch_id+1}st file batch")