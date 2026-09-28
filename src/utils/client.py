from src.models.semantic_codec import (
    VariationalDeepJSCC,
    VariationalDeepJSCCConfig,
)

def client_init_model(args):
    if args.algorithm == 'SCU':
        if args.dataset != 'cifar10':
            raise ValueError("SCU now supports CIFAR-10 only")
        return VariationalDeepJSCC(VariationalDeepJSCCConfig())
    raise ValueError("Client is only constructed for the single-process SCU baseline")

class Client:
    def __init__(self, data, private_test_data, args, rank):
        self.idx = rank
        self.args = args
        self.device = args.device
        self.data = data # Train Dataset
        self.model = client_init_model(args).to(self.device)
        
