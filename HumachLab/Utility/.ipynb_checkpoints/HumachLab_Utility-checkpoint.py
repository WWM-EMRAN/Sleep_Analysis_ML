"""
File Name: HumachLab_Utility.py
Author: Emran Ali
Email: wwm.emran@gmail.com, emran.ali@research.deakin.edu.au 
Date: 13/05/2021 5:54 am
"""


import os
import sys
import time
import datetime
import shutil
import numbers

### SRART: My modules ###
import HumachLab_Global
sys_ind = HumachLab_Global.sys_ind

if sys_ind==0:
    from HumachLab import *
elif sys_ind==1:
    from HumachLab import *
    from HumachLab.Utility.HumachLab_Logger import HumachLab_Logger
elif sys_ind==2:
    from HumachLab import *
    from HumachLab.Utility.HumachLab_Logger import HumachLab_Logger
    # import HumachLab.Utility.HumachLab_Logger as dm
    # sys.modules['HumachLab_Logger'] = dm
else:
    pass
### END: My modules ###


class Humachlab_Utility:
    def __init__(self):
        self._start_time = None
        self._end_time = None
        self.logger = None
        return


    # #################### LOGGER ####################
    # ### Creating and managing log directory, (experiment) serial number and general log file
    def create_log_path(self, machine_no, path, foldr, test_no, force_create=True):
        org_test_no = test_no
        srl_digits = 1000
        srl_series = machine_no*srl_digits
        srl = srl_series
        path2 = os.walk(path)
        for root, directories, files in path2:
            for dirs in directories:
                if len(dirs) > 0:
                    s = (dirs.split('_')[-1])
                    if s.isnumeric():
                        tmp_srl = int(s)
                        # print(tmp_srl, srl)
                        if ((tmp_srl>srl_series) and (tmp_srl<((machine_no+1)*srl_digits))) and (tmp_srl > srl):
                            srl = tmp_srl

        print(srl, test_no)
        srl = (srl+1)
        test_no = test_no + srl_series
        print(srl, test_no)
        srl = test_no if ((test_no%srl_digits)>0) else srl
        srl = f'{(srl):0{str(len(str(srl_digits)))}}'

        path = f'{path}{foldr}'
        if org_test_no>=0:
            path = f'{path}{srl}/'
        else:
            path = f'{path}/'

        msg_str = ''

        if force_create:
            if not os.path.exists(path):
                try:
                    os.mkdir(path)
                    msg = f'Directory created successfully: {path}\n'
                    msg_str += msg
                except:
                    msg = f'Can not create directory: {path}\n'
                    msg_str += msg
            else:
                msg = f'Result folder already exist: {path}\n'
                msg_str += msg
                print(msg)
                msg = f'Do you want to overwrite (y/n)? '
                ans = input(msg)
                msg += f'{ans}\n'
                msg_str += msg
                if ans=='Yes' or ans=='yes' or ans=='Y' or ans=='y':
                    msg = f'This will delete all the contents of the directory, Are you sure (y/n)? '
                    ans = input(msg)
                    msg += f'{ans}\n'
                    msg_str += msg

                    if ans == 'Yes' or ans == 'yes' or ans == 'Y' or ans == 'y':
                        try:
                            shutil.rmtree(path)
                            msg = f'Directory removed successfully: {path}\n'
                            msg_str += msg
                        except:
                            msg = f'Can not remove directory: {path}\n'
                            msg_str += msg

                        try:
                            os.mkdir(path)
                            msg = f'Directory re-created successfully: {path}\n'
                            msg_str += msg
                        except:
                            msg = f'Can not re-create directory: {path}\n'
                            msg_str += msg
                    else:
                        msg = f'Working with existing directory: {path}\n'
                        msg_str += msg

                else:
                    msg = f'Working with existing directory: {path}\n'
                    msg_str += msg
        else:
            msg_str = f'No need to create path: {path}\n'

        return path, srl, msg_str

    # ### Creating and managing log directory, (experiment) serial number and general log file
    def get_logger(self, logger_name, log_file_name):
        self.logger = (HumachLab_Logger(logger_name=logger_name, log_file_name=log_file_name)).logger
        return self.logger


    # #################### DATE AND TIME ####################
    def start_timer(self):
        self._start_time = time.time()
        return self._start_time

    def end_timer(self):
        self._end_time = time.time()
        return self._end_time

    # ### Calculating time and formatting it to display
    def time_calculator(self):
        total_time_str = ''
        if (self._start_time is None) or (self._end_time is None):
            total_time_str = f'<Start and end timers not set.>'
            return total_time_str
        st_time, en_time = self._start_time, self._end_time
        seconds = round(en_time-st_time, 2)
        conversion = datetime.timedelta(seconds=seconds)
        formatted_time = str(conversion)
        # print(seconds, formatted_time)

        if ('days' in formatted_time) or ('day' in formatted_time):
            number_of_days = int((formatted_time.split(',')[0].strip()).split(' ')[0].strip())
            years = number_of_days // 365
            total_time_str = (total_time_str + f'{years} Year(s) ') if years > 0 else ''
            months = (number_of_days - years * 365) // 30
            total_time_str = (total_time_str + f'{months:02} Month(s) ') if months > 0 or len(
                total_time_str) > 0 else ''
            days = (number_of_days - years * 365 - months * 30)
            total_time_str = (total_time_str + f'{days:02} Day(s) ') if days > 0 or len(total_time_str) > 0 else ''

            formatted_time = (formatted_time.split(',')[1]).strip()

        formatted_time = (formatted_time.split(':'))
        total_time_str = total_time_str + f'{formatted_time[0]} Hour(s) ' if int(formatted_time[0]) > 0 or len(
            total_time_str) > 0 else ''
        total_time_str = total_time_str + f'{formatted_time[1]} Minute(s) ' if int(formatted_time[1]) > 0 or len(
            total_time_str) > 0 else ''
        sec = round(float(formatted_time[2]), 2)
        total_time_str = total_time_str + f'{sec} Second(s) '

        self._start_time, self._end_time = None, None

        return total_time_str

