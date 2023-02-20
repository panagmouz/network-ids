import pandas as pd
import matplotlib.pyplot as plt


class UnswNb15:
    def __init__(self, testCase, isMulticlass=False):
        self.testCase = testCase
        self.isMulticlass = isMulticlass

        train = '.dataset_files/UNSW-NB15/UNSW_NB15_testing-set.csv'
        test = '.dataset_files/UNSW-NB15/UNSW_NB15_training-set.csv'

        feature_list = [
                'id', 'dur', 'proto', 'service', 'state', 'spkts', 'dpkts', 'sbytes',
                'dbytes', 'rate', 'sttl', 'dttl', 'sload', 'dload', 'sloss', 'dloss', 'sinpkt',
                'dinpkt', 'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat',
                'smean', 'dmean', 'trans_depth', 'response_body_len', 'ct_srv_src', 'ct_state_ttl', 'ct_dst_ltm',
                'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'is_ftp_login', 'ct_ftp_cmd',
                'ct_flw_http_mthd', 'ct_src_ltm', 'ct_srv_dst', 'is_sm_ips_ports', 'label', 'label_tmp'
            ]

        self.binary_flags = [
                'Fuzzers', 'Analysis', 'Backdoors', 'Backdoor', 'DoS', 'Exploits', 'Generic', 'Reconnaissance',
                'Shellcode', 'Worms'
            ]

        self.multiclass_flags = {
                'Backdoor': ['Backdoors'],
            }

        self.train_data = pd.read_csv(train, names=feature_list)
        self.test_data = pd.read_csv(test, names=feature_list)
        self.train_data['mode'] = 'Train'
        self.test_data['mode'] = 'Test'

    def graphs(self):
        plt.figure(figsize=(10, 7))
        self.test_data['protocol_type'].value_counts().plot(kind="bar")
        plt.xlabel('protocol_type')
        plt.ylabel('samples')
        plt.show()
        plt.figure(figsize=(10, 7))
        self.test_data['service'].value_counts().plot(kind="bar")
        plt.xlabel('service')
        plt.ylabel('samples')
        plt.show()
        plt.figure(figsize=(10, 7))
        self.test_data['flag'].value_counts().plot(kind="bar")
        plt.xlabel('flag')
        plt.ylabel('samples')
        plt.show()
        plt.figure(figsize=(10, 7))
        self.test_data['label'].value_counts().plot(kind="bar")
        plt.xlabel('label')
        plt.ylabel('samples')
        plt.show()

    def get(self):
        if self.testCase == 'test':
            data = pd.concat([self.train_data, self.test_data])
        else:
            print("undefined test")
            exit(-1)

        data['label'] = data['label'].replace(['Normal'], 0)

        if self.isMulticlass:
            for i in range(len(self.multiclass_flags['Backdoor'])):
                data['label'] = data['label'].replace([self.multiclass_flags['Backdoor'][i],
                                                       self.multiclass_flags['Backdoor'][i][:-1]], 1)
            data['label'] = data['label'].replace(['Fuzzers'], 2)
            data['label'] = data['label'].replace(['Analysis'], 3)
            data['label'] = data['label'].replace(['DoS'], 4)
            data['label'] = data['label'].replace(['Exploits'], 5)
            data['label'] = data['label'].replace(['Generic'], 6)
            data['label'] = data['label'].replace(['Reconnaissance'], 7)
            data['label'] = data['label'].replace(['Shellcode'], 8)
            data['label'] = data['label'].replace(['Worms'], 9)
        else:
            for i in range(len(self.binary_flags)):
                data['label'] = data['label'].replace(self.binary_flags[i], 1)

        tmp = data['proto'].copy()
        tmp = pd.get_dummies(tmp)
        data = data.drop(columns='proto', axis=1)
        data = pd.concat([data, tmp], axis=1)

        tmp = data['service'].copy()
        tmp = pd.get_dummies(tmp)
        data = data.drop(columns='service', axis=1)
        data = pd.concat([data, tmp], axis=1)

        tmp = data['state'].copy()
        tmp = pd.get_dummies(tmp)
        data = data.drop(columns='state', axis=1)
        data = pd.concat([data, tmp], axis=1)

        data = data.drop(columns='id', axis=1)
        data = data.drop(columns='label_tmp', axis=1)
        return data


if __name__ == "__main__":
    dataset = UnswNb15(testCase="test-21", isMulticlass=True)
    dataset.graphs()
    print(dataset.get())
