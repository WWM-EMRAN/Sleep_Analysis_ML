"""
File Name: HumachLab_Utility.py
Author: Emran Ali
Email: wwm.emran@gmail.com, emran.ali@research.deakin.edu.au 
Date: 13/05/2021 5:54 am
"""


class Humachlab_Utility:
    def __init__(self, log_path):
        self.log_path = log_path
        return

    # ### Save all information to logs
    # ### Temporary log
    def print_and_store_log(self, *all_tmp_data):
        tmp_data = ''
        num_arg = len(all_tmp_data)
        for i in range(num_arg):
            td = str(all_tmp_data[i])
            tmp_data += str(td) + ('\n' if (i==(num_arg-1) and (not td.endswith('\n'))) else ' ')

        print(tmp_data)
        self.temporary_log(tmp_data)
        return

    def temporary_log(self, tmp_data):
        # ### Save model scores and details
        log_path = self.log_path
        with open(log_path, 'a') as f:
            result = tmp_data
            f.write(result)
        return
    # ########


# ob = Humachlab_Utility('./test.txt')
# ob.print_and_store_log(1)
# ob.print_and_store_log(1, 7)
# ob.print_and_store_log('Hello')
# ob.print_and_store_log('Hi', 'there')
# ob.print_and_store_log([1, 3, 4, 5])
# ob.print_and_store_log({'a': 1, 'b':43, 'c':'time'})
