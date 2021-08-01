def is_inception_model(model):
    try:
        _ = model.AuxLogits.fc.in_features
    except:
        try:
            #model data parallel
            _ = model.module.AuxLogits.fc.in_features
        except:
            return False

    return True


