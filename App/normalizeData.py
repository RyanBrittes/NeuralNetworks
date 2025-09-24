import numpy as np

class NormalizeData():

    def calc_mean(self, raw_value):
        return np.mean(raw_value, axis=0)
    
    def calc_standard_deviation(self, raw_value):
        return np.std(raw_value, axis=0)
    
    def calc_score_Z(self, raw_value):
        mean_value = self.calc_mean(raw_value)
        std_value = self.calc_standard_deviation(raw_value)
        std_value = np.where(std_value == 0, 1, std_value)
        
        return (raw_value - mean_value) / std_value
    
    def calc_log(self, raw_value):
        list_value = []
        log_not_zero = 1e-14
        for i in range(len(raw_value)):
            list_value.append(np.log(raw_value[i] + log_not_zero))
        return np.array(list_value)
    
    def calc_log_denormalize(self, raw_value):
        return np.exp(raw_value)
    
    def calc_log_denormalize_list(self, raw_value):
        list_value = []
        for i in range(len(raw_value)):
            list_value.append(np.exp(raw_value[i]))

        return list_value
    