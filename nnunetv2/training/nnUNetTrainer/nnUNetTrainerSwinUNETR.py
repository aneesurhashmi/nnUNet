from monai.networks.nets import SwinUNETR
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerNoDeepSupervision import nnUNetTrainerNoDeepSupervision
import torch
import torch.nn as nn
from typing import Tuple, Union, List
# from nnunet_mednext import create_mednext_v1


class nnUNetTrainerSwinUNETR(nnUNetTrainerNoDeepSupervision):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                device: torch.device = torch.device('cuda')):
        """used for debugging plans etc"""
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 250
        
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = False) -> nn.Module:
        return SwinUNETR(
            img_size=(64, 160, 192),
            in_channels=num_input_channels,
            out_channels=num_output_channels,            
        )
        
        # return create_mednext_v1(
        #         num_channels = num_input_channels,
        #         num_classes = num_output_channels,
        #         model_id = 'B',             # S, B, M and L are valid model ids
        #         kernel_size = 3,            # 3x3x3 and 5x5x5 were tested in publication
        #         deep_supervision = enable_deep_supervision     # was used in publication
        #     )

