def get_feature_importance(model, features):

    if not hasattr(
        model,
        "feature_importances_"
    ):
        return None

    importance = model.feature_importances_

    return sorted(
        zip(features, importance),
        key=lambda x: x[1],
        reverse=True,
    )


def get_top_risk_drivers(
    model,
    features,
    top_n=3,
):

    importance = get_feature_importance(
        model,
        features
    )

    if importance is None:
        return []

    return importance[:top_n]