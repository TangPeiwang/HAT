import os
import torch
from models import HAT


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'HAT': HAT
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass

    def transfer_weights(self, weights_path, model, exclude_head=True, device='cpu'):
        # state_dict = model.state_dict()
        new_state_dict = torch.load(weights_path, map_location=device)
        # print('new_state_dict',new_state_dict)
        matched_layers = 0
        unmatched_layers = []
        for name, param in model.state_dict().items():
            if exclude_head and 'head' in name: continue
            if name in new_state_dict:
                matched_layers += 1
                input_param = new_state_dict[name]
                if input_param.shape == param.shape:
                    param.copy_(input_param)
                else:
                    unmatched_layers.append(name)
            else:
                unmatched_layers.append(name)
                pass  # these are weights that weren't in the original model, such as a new head
        if matched_layers == 0:
            raise Exception("No shared weight names were found between the models")
        else:
            if len(unmatched_layers) > 0:
                print(f'check unmatched_layers: {unmatched_layers}')
            else:
                print(f"weights from {weights_path} successfully transferred!\n")
        model = model.to(device)
        return model
