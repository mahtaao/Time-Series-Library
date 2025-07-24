def print_args(args):
    def safe_getattr(obj, attr, default='N/A'):
        return getattr(obj, attr, default)
    
    print("\033[1m" + "Basic Config" + "\033[0m")
    print(f'  {"Task Name:":<20}{safe_getattr(args, "task_name"):<20}{"Is Training:":<20}{safe_getattr(args, "is_training"):<20}')
    print(f'  {"Model ID:":<20}{safe_getattr(args, "model_id"):<20}{"Model:":<20}{safe_getattr(args, "model"):<20}')
    print()

    print("\033[1m" + "Data Loader" + "\033[0m")
    print(f'  {"Data:":<20}{safe_getattr(args, "data"):<20}{"Root Path:":<20}{safe_getattr(args, "root_path"):<20}')
    print(f'  {"Data Path:":<20}{safe_getattr(args, "data_path"):<20}{"Features:":<20}{safe_getattr(args, "features"):<20}')
    print(f'  {"Target:":<20}{safe_getattr(args, "target"):<20}{"Freq:":<20}{safe_getattr(args, "freq"):<20}')
    print(f'  {"Checkpoints:":<20}{safe_getattr(args, "checkpoints"):<20}')
    print()

    if safe_getattr(args, "task_name") in ['long_term_forecast', 'short_term_forecast']:
        print("\033[1m" + "Forecasting Task" + "\033[0m")
        print(f'  {"Seq Len:":<20}{safe_getattr(args, "seq_len"):<20}{"Label Len:":<20}{safe_getattr(args, "label_len"):<20}')
        print(f'  {"Pred Len:":<20}{safe_getattr(args, "pred_len"):<20}{"Seasonal Patterns:":<20}{safe_getattr(args, "seasonal_patterns"):<20}')
        print(f'  {"Inverse:":<20}{safe_getattr(args, "inverse"):<20}')
        print()

    if safe_getattr(args, "task_name") == 'imputation':
        print("\033[1m" + "Imputation Task" + "\033[0m")
        print(f'  {"Mask Rate:":<20}{safe_getattr(args, "mask_rate"):<20}')
        print()

    if safe_getattr(args, "task_name") == 'anomaly_detection':
        print("\033[1m" + "Anomaly Detection Task" + "\033[0m")
        print(f'  {"Anomaly Ratio:":<20}{safe_getattr(args, "anomaly_ratio"):<20}')
        print()

    print("\033[1m" + "Model Parameters" + "\033[0m")
    print(f'  {"Top k:":<20}{safe_getattr(args, "top_k"):<20}{"Num Kernels:":<20}{safe_getattr(args, "num_kernels"):<20}')
    print(f'  {"Enc In:":<20}{safe_getattr(args, "enc_in"):<20}{"Dec In:":<20}{safe_getattr(args, "dec_in"):<20}')
    print(f'  {"C Out:":<20}{safe_getattr(args, "c_out"):<20}{"d model:":<20}{safe_getattr(args, "d_model"):<20}')
    print(f'  {"n heads:":<20}{safe_getattr(args, "n_heads"):<20}{"e layers:":<20}{safe_getattr(args, "e_layers"):<20}')
    print(f'  {"d layers:":<20}{safe_getattr(args, "d_layers"):<20}{"d FF:":<20}{safe_getattr(args, "d_ff"):<20}')
    print(f'  {"Moving Avg:":<20}{safe_getattr(args, "moving_avg"):<20}{"Factor:":<20}{safe_getattr(args, "factor"):<20}')
    print(f'  {"Distil:":<20}{safe_getattr(args, "distil"):<20}{"Dropout:":<20}{safe_getattr(args, "dropout"):<20}')
    print(f'  {"Embed:":<20}{safe_getattr(args, "embed"):<20}{"Activation:":<20}{safe_getattr(args, "activation"):<20}')
    print()

    print("\033[1m" + "Run Parameters" + "\033[0m")
    print(f'  {"Num Workers:":<20}{safe_getattr(args, "num_workers"):<20}{"Itr:":<20}{safe_getattr(args, "itr"):<20}')
    print(f'  {"Train Epochs:":<20}{safe_getattr(args, "train_epochs"):<20}{"Batch Size:":<20}{safe_getattr(args, "batch_size"):<20}')
    print(f'  {"Patience:":<20}{safe_getattr(args, "patience"):<20}{"Learning Rate:":<20}{safe_getattr(args, "learning_rate"):<20}')
    print(f'  {"Des:":<20}{safe_getattr(args, "des"):<20}{"Loss:":<20}{safe_getattr(args, "loss"):<20}')
    print(f'  {"Lradj:":<20}{safe_getattr(args, "lradj"):<20}{"Use Amp:":<20}{safe_getattr(args, "use_amp"):<20}')
    print()

    print("\033[1m" + "GPU" + "\033[0m")
    print(f'  {"Use GPU:":<20}{safe_getattr(args, "use_gpu"):<20}{"GPU:":<20}{safe_getattr(args, "gpu"):<20}')
    print(f'  {"Use Multi GPU:":<20}{safe_getattr(args, "use_multi_gpu"):<20}{"Devices:":<20}{safe_getattr(args, "devices"):<20}')
    print()

    # Only print de-stationary projector params if they exist
    if hasattr(args, 'p_hidden_dims') and hasattr(args, 'p_hidden_layers'):
        print("\033[1m" + "De-stationary Projector Params" + "\033[0m")
        p_hidden_dims_str = ', '.join(map(str, args.p_hidden_dims))
        print(f'  {"P Hidden Dims:":<20}{p_hidden_dims_str:<20}{"P Hidden Layers:":<20}{args.p_hidden_layers:<20}') 
        print()
