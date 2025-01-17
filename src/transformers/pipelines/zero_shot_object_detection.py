import warnings
from typing import Any, Dict, List, Optional, Union

from ..utils import add_end_docstrings, is_torch_available, is_vision_available, logging, requires_backends
from ..utils.deprecation import deprecate_kwarg
from .base import Pipeline, build_pipeline_init_args


if is_vision_available():
    from PIL import Image

    from ..image_utils import load_image, valid_images

if is_torch_available():
    import torch

    from ..models.auto.modeling_auto import MODEL_FOR_ZERO_SHOT_OBJECT_DETECTION_MAPPING_NAMES

logger = logging.get_logger(__name__)


@add_end_docstrings(build_pipeline_init_args(has_processor=True))
class ZeroShotObjectDetectionPipeline(Pipeline):
    """
    Zero shot object detection pipeline using `OwlViTForObjectDetection`. This pipeline predicts bounding boxes of
    objects when you provide an image and a set of `candidate_labels`.

    Example:

    ```python
    >>> from transformers import pipeline

    >>> detector = pipeline(model="google/owlvit-base-patch32", task="zero-shot-object-detection")
    >>> detector(
    ...     "http://images.cocodataset.org/val2017/000000039769.jpg",
    ...     candidate_labels=["cat", "couch"],
    ... )
    [{'score': 0.287, 'label': 'cat', 'box': {'xmin': 324, 'ymin': 20, 'xmax': 640, 'ymax': 373}}, {'score': 0.254, 'label': 'cat', 'box': {'xmin': 1, 'ymin': 55, 'xmax': 315, 'ymax': 472}}, {'score': 0.121, 'label': 'couch', 'box': {'xmin': 4, 'ymin': 0, 'xmax': 642, 'ymax': 476}}]

    >>> detector(
    ...     "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png",
    ...     candidate_labels=["head", "bird"],
    ... )
    [{'score': 0.119, 'label': 'bird', 'box': {'xmin': 71, 'ymin': 170, 'xmax': 410, 'ymax': 508}}]
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

    This object detection pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"zero-shot-object-detection"`.

    See the list of available models on
    [huggingface.co/models](https://huggingface.co/models?filter=zero-shot-object-detection).
    """

    _load_processor = True
    _load_image_processor = False
    _load_feature_extractor = False
    _load_tokenizer = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self.framework == "tf":
            raise ValueError(f"The {self.__class__} is only available in PyTorch.")

        requires_backends(self, "vision")
        self.check_model_type(MODEL_FOR_ZERO_SHOT_OBJECT_DETECTION_MAPPING_NAMES)

    @deprecate_kwarg("images", version="4.51.0", new_name="inputs")
    @deprecate_kwarg("text_queries", version="4.51.0", new_name="candidate_labels")
    def __call__(
        self,
        inputs: Union[str, "Image.Image", List[Dict[str, Any]]],
        candidate_labels: Optional[Union[List[str], List[List[str]]]] = None,
        threshold: float = 0.1,
        top_k: Optional[int] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        """
        Detect objects (bounding boxes & classes) in the image(s) passed as inputs.

        Args:
            inputs (`str`, `PIL.Image` or `List[Dict[str, Any]]`):
                The pipeline handles three types of inputs:

                - A PIL image
                - A string containing an http url pointing to an image
                - A string containing a local path to an image

                You can also use this parameter to send directly a dataset, a generator or a list of images like so:

                - A dictionary in format {"image": image, "candidate_labels": candidate_labels}
                - A list of dictionaries in format [{"image": image, "candidate_labels": candidate_labels}, ...]

            candidate_labels (`str` or `List[str]` or `List[List[str]]`):
                What the model should recognize in the image.

            threshold (`float`, *optional*, defaults to 0.1):
                The probability necessary to make a prediction.

            top_k (`int`, *optional*, defaults to None):
                The number of top predictions that will be returned by the pipeline. If the provided number is `None`
                or higher than the number of predictions available, it will default to the number of predictions.

            timeout (`float`, *optional*, defaults to None):
                The maximum time in seconds to wait for fetching images from the web. If None, no timeout is set and
                the call may block forever.

        Example:

            ```python
            >>> from transformers import pipeline

            >>> detector = pipeline(model="google/owlvit-base-patch32", task="zero-shot-object-detection")
            >>> detector(
            ...     [
            ...         {
            ...             "image": "http://images.cocodataset.org/val2017/000000039769.jpg",
            ...             "candidate_labels": ["cat", "couch"],
            ...         },
            ...         {
            ...             "image": "http://images.cocodataset.org/val2017/000000039769.jpg",
            ...             "candidate_labels": ["cat", "couch"],
            ...         },
            ...     ]
            ... )
            [[{'score': 0.287, 'label': 'cat', 'box': {'xmin': 324, 'ymin': 20, 'xmax': 640, 'ymax': 373}}, {'score': 0.25, 'label': 'cat', 'box': {'xmin': 1, 'ymin': 55, 'xmax': 315, 'ymax': 472}}, {'score': 0.121, 'label': 'couch', 'box': {'xmin': 4, 'ymin': 0, 'xmax': 642, 'ymax': 476}}], [{'score': 0.287, 'label': 'cat', 'box': {'xmin': 324, 'ymin': 20, 'xmax': 640, 'ymax': 373}}, {'score': 0.254, 'label': 'cat', 'box': {'xmin': 1, 'ymin': 55, 'xmax': 315, 'ymax': 472}}, {'score': 0.121, 'label': 'couch', 'box': {'xmin': 4, 'ymin': 0, 'xmax': 642, 'ymax': 476}}]]
            ```


        Return:
            A list of lists containing prediction results, one list per input image. Each list contains dictionaries
            with the following keys:

            - **label** (`str`) -- Text query corresponding to the found object.
            - **score** (`float`) -- Score corresponding to the object (between 0 and 1).
            - **box** (`Dict[str, int]`) -- Bounding box of the detected object in image's original size. It is a
              dictionary with `x_min`, `x_max`, `y_min`, `y_max` keys.
        """

        # Case 1. Single image + list of candidate labels (list of strings)
        if isinstance(inputs, (str, Image.Image)):
            if not isinstance(candidate_labels, (list, tuple)):
                raise ValueError(f"`candidate_labels` should be a list of strings, got `{type(candidate_labels)}`")

            standardized_inputs = [{"image": inputs, "candidate_labels": candidate_labels}]

        # Case 2. List of images + list of candidate labels (list of lists of strings)
        elif isinstance(inputs, (list, tuple)) and valid_images(inputs):
            if (
                not isinstance(candidate_labels, (list, tuple))
                or not all(isinstance(image_labels, (list, tuple)) for image_labels in candidate_labels)
                or not len(inputs) == len(candidate_labels)
            ):
                raise ValueError(
                    "`candidate_labels` should be a list of lists of strings with the same length as `inputs`"
                )

            standardized_inputs = [
                {"image": image, "candidate_labels": labels} for image, labels in zip(inputs, candidate_labels)
            ]

        # Case 3. Supports the following format
        #  - {"image": image, "candidate_labels": candidate_labels}
        #  - [{"image": image, "candidate_labels": candidate_labels}]
        #  - Generator and datasets
        # This is a common pattern in other multimodal pipelines, so we support it here as well.
        else:
            if candidate_labels is not None:
                raise ValueError(
                    "Expecting `candidate_labels` to be a part of `inputs` and not passed as a separate argument. "
                    "For example, `result = pipe(inputs=[{'image': image, 'candidate_labels': candidate_labels}, ...])`"
                )
            standardized_inputs = inputs

        results = super().__call__(standardized_inputs, timeout=timeout, threshold=threshold, top_k=top_k, **kwargs)

        return results

    def _sanitize_parameters(self, **kwargs):
        """Split input __call__ kwargs subsets for preprocessing, forward and postprocessing."""

        preprocessing_keys = ["timeout"]
        postprocessing_keys = [
            "threshold",
            "top_k",
            "nms_threshold",  # Omdet Turbo
            "text_threshold",  # Grounding DINO
        ]

        preprocessing_kwargs = {k: kwargs.pop(k) for k in preprocessing_keys if k in kwargs}
        postprocessing_kwargs = {k: kwargs.pop(k) for k in postprocessing_keys if k in kwargs}

        if kwargs:
            warnings.warn(f"The following kwargs were ignored by the pipeline: {kwargs.keys()}")
        return preprocessing_kwargs, {}, postprocessing_kwargs

    def preprocess(self, inputs: Dict[str, Any], timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Preprocess the inputs with Processor class.

        Args:
            inputs (Dict[str, Any]):
                The inputs to preprocess. Always a single sample, iteration is handled by the pipeline.
            timeout (Optional[float]):
                The timeout for the image to be loaded in case URL is provided.

        Returns:
            Dict[str, Any]: The preprocessed inputs.
        """

        image = load_image(inputs["image"], timeout=timeout)
        candidate_labels = inputs["candidate_labels"]

        model_inputs = self.processor(
            images=image,
            text=candidate_labels,
            return_tensors=self.framework,
        )
        model_inputs["pixel_values"] = model_inputs["pixel_values"].to(self.torch_dtype)

        target_sizes = [[image.height, image.width]]
        target_sizes = torch.tensor(target_sizes, dtype=torch.int32)

        return {
            # For postprocessing
            "target_sizes": target_sizes,
            "candidate_labels": candidate_labels,
            # The preprocessed inputs will be collated using self.collate_fn, which requires
            # tensors to be at the root level of the dictionary.
            **model_inputs,
        }

    def _forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Forward the preprocessed (by self.preprocess) and collated (by self.collate)
        batch of inputs to the model.

        Args:
            inputs (Dict[str, Any]):
                The inputs to forward.

        Returns:
            Dict[str, Any]: The outputs of the model.
        """

        # To avoid passing unnecessary arguments into the model forward
        target_sizes = inputs.pop("target_sizes")
        candidate_labels = inputs.pop("candidate_labels")
        batch_size = len(target_sizes)

        model_outputs = self.model(**inputs)

        # We convert ModelOutput to List[ModelOutput, ...] wrapping each sample
        # to avoid it's conversion in PipelineIterator. PipelineIterator call .as_tuple(),
        # but we need to keep the original class to be able to pass output to postprocessing method
        # of the processor.
        ModelOutputClass = type(model_outputs)
        model_outputs_list = []
        for i in range(batch_size):
            # slice instead of indexing to preserve batch dimension
            data = {k: v[i : i + 1] for k, v in model_outputs.items()}
            model_outputs_list.append(ModelOutputClass(**data))

        return {
            "target_sizes": target_sizes,
            "candidate_labels": candidate_labels,
            "model_outputs": model_outputs_list,
        }

    def postprocess(
        self, output, threshold: float = 0.1, top_k: Optional[int] = None, **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Apply postprocessing to the model outputs to get the final predictions.
        Always called for a single sample.

        Args:
            output (`ModelOutput`):
                Model specific output object containing the model outputs like logits, hidden states, etc.
            threshold (`float`, *optional*, defaults to 0.1):
                The probability necessary to keep a prediction based on confidence score.
            tok_k (`int`, *optional*, defaults to None):
                The number of top predictions that will be returned by the pipeline. If the provided number is `None`
                or higher than the number of predictions available, it will default to the number of predictions.


        Returns:


        """
        # it's a list like ["cat", "dog"], wrap for batch of one sample
        candidate_labels = [output["candidate_labels"]]

        postprocessed_outputs = self.processor.post_process_grounded_object_detection(
            outputs=output["model_outputs"],
            target_sizes=output["target_sizes"],
            text_labels=candidate_labels,
            threshold=threshold,
            **kwargs,
        )

        # `postprocess` always get a batch of exactly one sample
        postprocessed_output = postprocessed_outputs[0]

        # Convert to pipeline format
        results = []
        for score, text_label, box in zip(
            postprocessed_output["scores"],
            postprocessed_output["text_labels"],
            postprocessed_output["boxes"],
        ):
            score = score.item()
            box = self._get_bounding_box(box)
            result = {"score": score, "label": text_label, "box": box}
            results.append(result)

        # Sort by score and keep top_k
        results = sorted(results, key=lambda x: x["score"], reverse=True)
        if top_k:
            results = results[:top_k]

        return results

    def _get_bounding_box(self, box: "torch.Tensor") -> Dict[str, int]:
        """
        Turns list [xmin, xmax, ymin, ymax] into dict { "xmin": xmin, ... }

        Args:
            box (`torch.Tensor`): Tensor containing the coordinates in corners format.

        Returns:
            bbox (`Dict[str, int]`): Dict containing the coordinates in corners format.
        """
        if self.framework != "pt":
            raise ValueError("The ZeroShotObjectDetectionPipeline is only available in PyTorch.")
        xmin, ymin, xmax, ymax = box.int().tolist()
        bbox = {
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
        }
        return bbox
