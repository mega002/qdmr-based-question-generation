import torch


def share_model_weights_by_fqn(model, source_names, target_names, tie=False):
    for source_name, target_name in zip(source_names, target_names):
        source_module = find_module_by_name(model, source_name)
        target_module = find_module_by_name(model, target_name)

        share_modules_weights(source_module, target_module, tie)


def find_module_by_name(model, name):
    levels = name.split(".")
    module = model
    for level in levels:
        module = module._modules[level]
    return module


def share_modules_weights(source_module, target_module, tie=False):
    assert type(source_module) == type(target_module)

    for parameter in source_module._parameters:
        if parameter in target_module._parameters:
            if tie:
                target_module._parameters[parameter] = source_module._parameters[
                    parameter
                ]
            else:
                target_module._parameters[parameter] = torch.nn.Parameter(
                    source_module._parameters[parameter].clone()
                )
    for submodule in source_module._modules:
        if submodule in target_module._modules:
            share_modules_weights(
                source_module._modules[submodule],
                target_module._modules[submodule],
                tie,
            )
