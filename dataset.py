from datasets.Brain4cars import Brain4cars


def get_training_set(opt, spatial_transform, horizontal_flip, temporal_transform,
                     target_transform):
    assert opt.dataset in ['Brain4cars']

    training_data = Brain4cars(
            opt.video_path,
            opt.annotation_path,
            'training',
            opt.n_fold,
            opt.end_second,
            1,
            spatial_transform=spatial_transform,
            horizontal_flip=horizontal_flip,
            temporal_transform=temporal_transform,
            target_transform=target_transform)

    return training_data


def get_validation_set(opt, spatial_transform, temporal_transform,
                       target_transform):
    assert opt.dataset in ['Brain4cars']

    validation_data = Brain4cars(
            opt.video_path,
            opt.annotation_path,
            'validation',
            opt.n_fold,
            opt.end_second,
            opt.n_val_samples,
            spatial_transform,
            None,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration)
    return validation_data
