import pandas as pd
import matplotlib.pyplot as plt


class Kdd:
    def __init__(self, testCase, isMulticlass=False):
        self.testCase = testCase
        self.isMulticlass = isMulticlass

        train = '.dataset_files/KDD99/training'
        test = '.dataset_files/KDD99/testing'

        feature_list = [
            "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment",
            "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted",
            "num_root", "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
            "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
            "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
            "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
            "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
            "dst_host_srv_rerror_rate", "label"
        ]

        self.binary_flags = [
            'ipsweep.', 'nmap.', 'portsweep.', 'satan.', 'saint.', 'mscan.', 'back.', 'land.', 'neptune.', 'pod.',
            'smurf.', 'teardrop.', 'apache2.', 'udpstorm.', 'processtable.', 'mailbomb.', 'buffer_overflow.',
            'loadmodule.', 'perl.', 'rootkit.', 'xterm.', 'ps.', 'sqlattack.', 'ftp_write.', 'guess_passwd.',
            'imap.', 'multihop.', 'phf.', 'spy.', 'warezclient.', 'warezmaster.', 'snmpgetattack.', 'named.',
            'xlock.', 'xsnoop.', 'sendmail.', 'httptunnel.', 'worm.', 'snmpguess.'
        ]

        self.multiclass_flags = {
            'normal': 'normal',
            'probe': ['ipsweep.', 'nmap.', 'portsweep.', 'satan.', 'saint.', 'mscan.'],
            'dos': ['back.', 'land.', 'neptune.', 'pod.', 'smurf.', 'teardrop.', 'apache2.', 'udpstorm.',
                    'processtable.', 'mailbomb.'],
            'u2r': ['buffer_overflow.', 'loadmodule.', 'perl.', 'rootkit.', 'xterm.', 'ps.', 'sqlattack.'],
            'r2l': ['ftp_write.', 'guess_passwd.', 'imap.', 'multihop.', 'phf.', 'spy.', 'warezclient.', 'warezmaster.',
                    'snmpgetattack.', 'named.', 'xlock.', 'xsnoop.', 'sendmail.', 'httptunnel.', 'worm.', 'snmpguess.']
        }

        self.samples_to_fill = [0]
        self.samples_encoding = [[1, 0, 0, 0, 0]]

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

        data['label'] = data['label'].replace(['normal.', 'normal'], 0)

        if self.isMulticlass:
            for i in range(len(self.multiclass_flags['probe'])):
                data['label'] = data['label'].replace(
                    [self.multiclass_flags['probe'][i], self.multiclass_flags['probe'][i][:-1]], 1)
            for i in range(len(self.multiclass_flags['dos'])):
                data['label'] = data['label'].replace(
                    [self.multiclass_flags['dos'][i], self.multiclass_flags['dos'][i][:-1]], 2)
            for i in range(len(self.multiclass_flags['u2r'])):
                data['label'] = data['label'].replace(
                    [self.multiclass_flags['u2r'][i], self.multiclass_flags['u2r'][i][:-1]], 3)
            for i in range(len(self.multiclass_flags['r2l'])):
                data['label'] = data['label'].replace(
                    [self.multiclass_flags['r2l'][i], self.multiclass_flags['r2l'][i][:-1]], 4)
        else:
            for i in range(len(self.binary_flags)):
                data['label'] = data['label'].replace(self.binary_flags[i], 1)

        tmp = data['protocol_type'].copy()
        tmp = pd.get_dummies(tmp)
        data = data.drop(columns='protocol_type', axis=1)
        data = pd.concat([data, tmp], axis=1)

        tmp = data['service'].copy()
        tmp = pd.get_dummies(tmp)
        data = data.drop(columns='service', axis=1)
        data = pd.concat([data, tmp], axis=1)

        tmp = data['flag'].copy()
        tmp = pd.get_dummies(tmp)
        data = data.drop(columns='flag', axis=1)
        data = pd.concat([data, tmp], axis=1)

        return data


if __name__ == "__main__":
    dataset = Kdd(testCase="test", isMulticlass=True)
    dataset.graphs()
    print(dataset.get())
