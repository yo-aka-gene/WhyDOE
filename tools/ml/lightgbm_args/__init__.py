def multiclass_args(
    num_class: int,
    objective="multiclass", 
    metric="multi_logloss",
    verbosity=-1, 
    deterministic=True,
    random_seed=0, 
    num_boost_round=100,
    force_col_wise=True
) -> dict:
    return dict(
        objective=objective, 
        metric=metric,
        num_class=num_class,
        verbosity=verbosity, 
        deterministic=deterministic,
        random_seed=random_seed, 
        num_boost_round=num_boost_round,
        force_col_wise=force_col_wise
    )


def regression_args(
    objective="regression",
    metric="l2",
    verbosity=-1,
    deterministic=True,
    random_seed=0, 
    num_boost_round=100,
    force_col_wise=True
) -> dict:
    return dict(
        objective=objective,
        metric=metric,
        verbosity=verbosity,
        deterministic=deterministic,
        random_seed=random_seed, 
        num_boost_round=num_boost_round,
        force_col_wise=force_col_wise
    )


__all__ = [
    multiclass_args,
    regression_args,
]