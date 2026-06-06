def get_model(model_name, args):
    name = model_name.lower()
    if name == "icarl":
        from models.icarl import iCaRL
        return iCaRL(args)
    elif name == "icarl_tal":
        from models.icarl_tal import iCaRLTAL
        return iCaRLTAL(args)
    elif name == "der":
        from models.der import DER
        return DER(args)
    elif name == "der_tal":
        from models.der_tal import DER
        return DER(args)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
