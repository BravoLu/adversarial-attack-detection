import argparse 

class Option(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='adversarial attack detection')
        self.parser.add_argument('--gpu', type=str, default='0')
        

