# -*- coding: utf-8 -*

from pprint import pprint

class Config(object):
    arch_name = 'resnet18'
    feature_layer_name = 'avgpool' # Module for extracting features in the middle
    feature_index_in_module = 0 # Which layer feature to use in Module
    class_nums = -1 # class_nums < 0 not reset class_nums, class_nums > 0 reset
    input_size = 224
    pretrained = True
    checkpoint = ''
    batch_size = 2
    batch_images_path = " "
    single_images = " "


    def _parse(self, kwargs):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            setattr(self, k, v)

        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items() if not k.startswith('_')}