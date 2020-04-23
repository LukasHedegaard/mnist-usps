from pathlib import Path
import datasetops as do
from torchvision.datasets import MNIST, USPS

DATA_PATH = Path(__file__).parent.parent.parent / "data"

domains = {"amazon", "dslr", "webcam"}


def mnistusps(
    source_name="mnist",
    target_name="usps",
    seed=1,
    num_source_per_class=200,
    num_target_per_class=3,
    same_to_diff_class_ratio=3,
    image_resize=None,
    group_in_out=False,
    framework_conversion=None,
    data_path=None,
):
    """Get a train-val-test split for Office 31 domain adaptation experiments

    Keyword Arguments:
        source_name {str} -- one of {'amazon','dslr','webcam'}
        target_name {str} -- one of {'amazon','dslr','webcam'}
        seed {int} -- random seed for train-val split
        num_source_per_class {int} -- number of source samples per clas,
        num_target_per_class {int} -- number of target samples per class,
        same_to_diff_class_ratio {int} -- ratio of same-class to different-class sample pairs in the train set
        image_resize {Tuple[int,int]} -- size to resize images to. If `None`, a path is returned
        group_in_out {bool} -- group images and labels separately as input and output tuples
        framework_conversion {Optional[str]} -- framework to convert to. Options: ["tensorflow", "pytorch"]
        data_path {Union[str, Path, None]} -- path to data. If no path is given, the data will we automatically downloaded into the "~/data" folder
        seed {int} -- random seed to be used for val-test split (default: {1})

    Returns:
        Tuple[Dataset] -- train, val and test datasetops.Dataset objects. To use these with Tensorflow or Pytorch, use the built-in `to_tensorflow` or `to_pytorch` methods.
    """
    data_path = Path(data_path) if data_path else DATA_PATH

    def get_data(domain_name: str, split="train"):
        train = split == "train"
        if domain_name == "mnist":
            return do.from_pytorch(MNIST(data_path, download=True, train=train))
        if domain_name == "usps":
            return do.from_pytorch(USPS(data_path, download=True, train=train))

    source = get_data(source_name, "train").named("s_data", "s_label")
    target_trainval = get_data(target_name, "train").named("t_data", "t_label")
    target_test = get_data(target_name, "test").named("t_data", "t_label")

    source_train = source.shuffle(seed).filter(
        s_label=do.allow_unique(num_source_per_class)
    )

    target_train, target_val = target_trainval.shuffle(seed).split_filter(
        t_label=do.allow_unique(num_target_per_class)
    )

    # Pair up all combinations of the datasets:
    # [(sx1, sy1, tx1, ty1), (sx1, sy1, tx2, ty2) ... ]
    train_cart = do.cartesian_product(source_train, target_train)

    # Limit the train set to have at most an 1:3 ratio of same- and diff-label pairs
    train_same, train_diff = train_cart.reorder(
        "s_data", "t_data", "s_label", "t_label"
    ).split_filter(lambda x: x[2] == x[3])

    if len(train_diff) > same_to_diff_class_ratio * len(train_same):
        train_diff = train_diff.sample(3 * len(train_same), seed=seed)

    train = do.concat(train_same, train_diff).shuffle(seed)

    # Pair each datapoint with itself, (x,x,y,y)
    val, test = [do.zipped(d, d).reorder(0, 2, 1, 3) for d in [target_val, target_test]]

    if image_resize:
        train, val, test = [
            ds.image_resize(image_resize, image_resize) for ds in [train, val, test]
        ]

    if group_in_out:
        # Change the data representation into two tuples (in, out) with extra out label
        train, val, test = [
            ds.transform(lambda x: ((x[0], x[1]), (x[2], x[3])))
            for ds in [train, val, test]
        ]

    if framework_conversion == "tensorflow":
        train, val, test = [ds.to_tensorflow() for ds in [train, val, test]]
    elif framework_conversion == "pytorch":
        train, val, test = [ds.to_pytorch() for ds in [train, val, test]]

    return train, val, test
