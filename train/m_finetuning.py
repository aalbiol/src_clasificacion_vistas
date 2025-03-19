from pytorch_lightning.callbacks import BaseFinetuning

def count_unfrozen_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class FeatureExtractorFreezeUnfreeze(BaseFinetuning):
    def __init__(self, unfreeze_at_epoch=10,initial_denom_lr=2,freeze_conditioner=False):
        super().__init__()
        self._unfreeze_at_epoch = unfreeze_at_epoch
        self.initial_denom_lr=initial_denom_lr
        self.freeze_conditioner=freeze_conditioner

    def freeze_before_training(self, pl_module):
        # freeze any module you want
        # Here, we are freezing `feature_extractor`
        self.freeze(pl_module.modelo.features)
        if self.freeze_conditioner:
            print("Freezing conditioner")
            self.freeze(pl_module.modelo.conditioner)

    def finetune_function(self, pl_module, current_epoch, optimizer):
    # When `current_epoch` is 10, feature_extractor will start training.
        submodulos=list(pl_module.modelo.features.children())
        if current_epoch == self._unfreeze_at_epoch:
            self.unfreeze_and_add_param_group(
            #modules=pl_module.modelo.features.layer4,
            modules=submodulos[-1],
            optimizer=optimizer,
            train_bn=True,
            initial_denom_lr=self.initial_denom_lr
            )
            unfrozen_params=count_unfrozen_parameters(pl_module.modelo)
            print(f"Unfreezing layer4. Learning {unfrozen_params} parameters")
            
        if current_epoch == 2*self._unfreeze_at_epoch:
            self.unfreeze_and_add_param_group(
            #modules=pl_module.modelo.features.layer3,
            modules=submodulos[-2],
            optimizer=optimizer,
            train_bn=True,
            initial_denom_lr=self.initial_denom_lr
            )
            unfrozen_params=count_unfrozen_parameters(pl_module.modelo)
            print(f"Unfreezing layer3. Learning {unfrozen_params} parameters")            

        if current_epoch == 3*self._unfreeze_at_epoch:
            self.unfreeze_and_add_param_group(
            #modules=pl_module.modelo.features.layer2,
            modules=submodulos[-3],
            optimizer=optimizer,
            train_bn=True,
            initial_denom_lr=self.initial_denom_lr
            )
            unfrozen_params=count_unfrozen_parameters(pl_module.modelo)
            print(f"Unfreezing layer2. Learning {unfrozen_params} parameters") 
            
        if current_epoch == 4*self._unfreeze_at_epoch:
            self.unfreeze_and_add_param_group(
            #modules=pl_module.modelo.features.layer2,
            modules=submodulos[-4],
            optimizer=optimizer,
            train_bn=True,
            initial_denom_lr=self.initial_denom_lr
            )
            unfrozen_params=count_unfrozen_parameters(pl_module.modelo)
            print(f"Unfreezing layer1. Learning {unfrozen_params} parameters")                                 