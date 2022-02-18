class Logging():
    def __init__(self, save_path):
        self.filename = save_path + "/result.log"

    def record(self, str_log):
        print(str_log)
        with open(self.filename, 'a') as f:
            f.write("%s\r" % str_log)
            f.flush()
