"""
File Name: HumachLab_self.logger.py 
Author: Emran Ali
Email: wwm.emran@gmail.com, emran.ali@research.deakin.edu.au 
Date: 13/05/2021 6:20 pm
"""

import logging
import sys

# logger = logging.getLogger()

class HumachLab_Logger:
    # def __init__(self, logger_name=__name__, log_file_name='./log.txt', log_format='%(asctime)s - %(name)s: %(levelname)s - '\
    #                                                                                '%(module)s - %(funcName)s() - %(lineno)d: \n%(message)s'):
    def __init__(self, logger_name=__name__, log_file_name='./log.txt', log_type=logging.INFO):

        logind = log_file_name.rfind('_')
        log_file_name_detail = f'{log_file_name[:logind]}_detail{log_file_name[logind:]}'
        log_file_name_detail_dict = f'{log_file_name[:logind]}_detail_dict{log_file_name[logind:]}'

        log_format = '%(message)s'
        log_format_detail = "%(asctime)s - [%(name)s]: [%(levelname)s] - %(module)s - %(funcName)s() - %(lineno)d \n%(message)s"
        log_format_detail_dict = "{'time':'%(asctime)s', 'loggername':'%(name)s', 'logtype':'%(levelname)s', 'module':'%(module)s',  " \
                                 "'function':'%(funcName)s()', 'linenumber':%(lineno)d: 'message':'%(message)s'}"

        # global logger
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.DEBUG) #DEBUG, INFO, WORNING, ERROR, CRITICAL, (logger.exception)

        # ### Log to message only log
        formatter = logging.Formatter(log_format)
        
        file_handler = logging.FileHandler(log_file_name, mode='a')
        # file_handler.setLevel(logging.ERROR)
        file_handler.setLevel(log_type)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # ### Log to standard output
        # # Log as error
        # stream_handler = logging.StreamHandler()
        # # Log as output
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)
        # stream_handler2 = logging.StreamHandler(sys.stderr)
        # stream_handler2.setFormatter(formatter)
        # self.logger.addHandler(stream_handler2)

        # ### Log to detailed log
        formatter_detail = logging.Formatter(log_format_detail)

        file_handler_detail = logging.FileHandler(log_file_name_detail, mode='a')
        # file_handler.setLevel(logging.ERROR)
        file_handler_detail.setLevel(logging.INFO)
        file_handler_detail.setFormatter(formatter_detail)
        self.logger.addHandler(file_handler_detail)

        # ### Log to detailed dictionary log
        formatter_detail_dict = logging.Formatter(log_format_detail_dict)

        file_handler_detail_dict = logging.FileHandler(log_file_name_detail_dict, mode='a')
        # file_handler.setLevel(logging.ERROR)
        file_handler_detail_dict.setLevel(logging.INFO)
        file_handler_detail_dict.setFormatter(formatter_detail_dict)
        self.logger.addHandler(file_handler_detail_dict)


        return
    
    
    
