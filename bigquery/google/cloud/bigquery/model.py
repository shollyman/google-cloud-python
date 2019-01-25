# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Define API Models."""

from __future__ import absolute_import

import copy
import datetime
import json
import operator
import warnings

import six

try:
    import pandas
except ImportError:  # pragma: NO COVER
    pandas = None

from google.api_core.page_iterator import HTTPIterator

import google.cloud._helpers
from google.cloud.bigquery import _helpers


_NO_PANDAS_ERROR = (
    "The pandas library is not installed, please install "
    "pandas to use the to_dataframe() function."
)
_TABLE_HAS_NO_SCHEMA = 'Table has no schema:  call "client.get_table()"'
_MARKER = object()


def _reference_getter(model):
    """A :class:`~google.cloud.bigquery.model.ModelReference` pointing to
    this model.

    Returns:
        google.cloud.bigquery.model.ModelReference: pointer to this model.
    """
    from google.cloud.bigquery import dataset

    dataset_ref = dataset.DatasetReference(model.project, model.dataset_id)
    return ModelReference(dataset_ref, model.model_id)


class ModelReference(object):
    """ModelReferences are pointers to models.

    See
    https://cloud.google.com/bigquery/docs/reference/rest/v2/models

    Args:
        dataset_ref (google.cloud.bigquery.dataset.DatasetReference):
            A pointer to the dataset
        model_id (str): The ID of the model
    """

    def __init__(self, dataset_ref, model_id):
        self._project = dataset_ref.project
        self._dataset_id = dataset_ref.dataset_id
        self._model_id = model_id

    @property
    def project(self):
        """str: Project bound to the model"""
        return self._project

    @property
    def dataset_id(self):
        """str: ID of dataset containing the model."""
        return self._dataset_id

    @property
    def model_id(self):
        """str: The model ID."""
        return self._model_id

    @property
    def path(self):
        """str: URL path for the model's APIs."""
        return "/projects/%s/datasets/%s/models/%s" % (
            self._project,
            self._dataset_id,
            self._model_id,
        )

    @classmethod
    def from_string(cls, model_id, default_project=None):
        """Construct a model reference from model ID string.

        Args:
            model_id (str):
                A model ID in standard SQL format. If ``default_project``
                is not specified, this must included a project ID, dataset
                ID, and model ID, each separated by ``.``.
            default_project (str):
                Optional. The project ID to use when ``model_id`` does not
                include a project ID.

        Returns:
            ModelReference: Model reference parsed from ``model_id``.

        Examples:
            >>> ModelReference.from_string('my-project.mydataset.mymodel')
            ModelRef...(DatasetRef...('my-project', 'mydataset'), 'mymodel')

        Raises:
            ValueError:
                If ``model_id`` is not a fully-qualified model ID in
                standard SQL format.
        """
        from google.cloud.bigquery.dataset import DatasetReference

        output_project_id = default_project
        output_dataset_id = None
        output_model_id = None
        parts = model_id.split(".")

        if len(parts) < 2:
            raise ValueError(
                "model_id must be a fully-qualified model ID in "
                'standard SQL format. e.g. "project.dataset.model", got '
                "{}".format(model_id)
            )
        elif len(parts) == 2:
            if not default_project:
                raise ValueError(
                    "When default_project is not set, model_id must be a "
                    "fully-qualified model ID in standard SQL format. "
                    'e.g. "project.dataset_id.model_id", got {}'.format(model_id)
                )
            output_dataset_id, output_model_id = parts
        elif len(parts) == 3:
            output_project_id, output_dataset_id, output_model_id = parts
        if len(parts) > 3:
            raise ValueError(
                "Too many parts in model_id. Must be a fully-qualified model "
                'ID in standard SQL format. e.g. "project.dataset.model", '
                "got {}".format(model_id)
            )

        return cls(
            DatasetReference(output_project_id, output_dataset_id), output_model_id
        )

    @classmethod
    def from_api_repr(cls, resource):
        """Factory:  construct a model reference given its API representation

        Args:
            resource (Dict[str, object]):
                Model reference representation returned from the API

        Returns:
            google.cloud.bigquery.model.ModelReference:
                Model reference parsed from ``resource``.
        """
        from google.cloud.bigquery.dataset import DatasetReference

        project = resource["projectId"]
        dataset_id = resource["datasetId"]
        model_id = resource["modelId"]
        return cls(DatasetReference(project, dataset_id), model_id)

    def to_api_repr(self):
        """Construct the API resource representation of this model reference.

        Returns:
            Dict[str, object]: Model reference represented as an API resource
        """
        return {
            "projectId": self._project,
            "datasetId": self._dataset_id,
            "modelId": self._model_id,
        }

    def _key(self):
        """A tuple key that uniquely describes this field.

        Used to compute this instance's hashcode and evaluate equality.

        Returns:
            Tuple[str]: The contents of this :class:`DatasetReference`.
        """
        return (self._project, self._dataset_id, self._model_id)

    def __eq__(self, other):
        if not isinstance(other, ModelReference):
            return NotImplemented
        return self._key() == other._key()

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(self._key())

    def __repr__(self):
        from google.cloud.bigquery.dataset import DatasetReference

        dataset_ref = DatasetReference(self._project, self._dataset_id)
        return "ModelReference({}, '{}')".format(repr(dataset_ref), self._model_id)

class TrainingRun(object):
    """TrainingRuns contains statistics for a single training run of a model.

    See
    https://cloud.google.com/bigquery/docs/reference/rest/v2/models
    """

    def __init__(self):
        self._properties = {}

    @classmethod
    def from_api_repr(cls, resource):
        """Factory:  construct a TrainingRun given its API representation.

        Args:
            resource (Dict[str, object])
                TrainingRun representation returned from the API

        Returns:
            google.cloud.bigquery.model.TrainingRun:
                TrainingRun parsed from ``resource``.
        """
        training_run = cls()
        training_run._properties = resource
        return training_run

    def to_api_repr(self):
        """Constructs the API resource of this TrainingRun

        Returns:
            Dict[str, object]: TrainingRun represented as an API resource.
        """
        return copy.deepcopy(self._properties)

    @property
    def training_options(self):
        """google.cloud.bigquery.model.TrainingOptions: Training Options 
        specified for the run.
        """
        prop = self._properties.get("trainingOptions")

    @property
    def started(self):
        """Union[datetime.datetime, None]: Datetime at which the TrainingRun was started.
        """
        start_time = self._properties.get("startTime")
        if start_time is not None:
            return google.cloud._helpers._datetime_from_microseconds(
                1000.0 * float(start_time)
            )
    
    @property
    def evaluation_metrics(self):
        # TODO: 
        #       EvaluationMetrics
        #           RegressionMetrics
        #           BinaryClassificationMetrics
        #           AggregateClassificationMetrics
        #           BinaryConfusionMetrics
        #               ConfusionMatrix
        #                   Row
        #                   Entry
        #           MultiClassClassificationMetrics
        raise NotImplementedError


class TrainingOptions(object):
    """TrainingOptions represent options for a given training run.
    
    See
    https://cloud.google.com/bigquery/docs/reference/rest/v2/models
    """

    @classmethod
    def from_api_repr(cls, resource):
        """Factory:  construct a TrainingOptions given its API representation.

        Args:
            resource (Dict[str, object])
                TrainingOptions representation returned from the API.

        Returns:
            google.cloud.bigquery.model.TrainingOptions:
                TrainingOptions parsed from ``resource``.
        """
        training_options = cls(TrainingOptions())
        training_options._properties = resource

        return training_run

    def to_api_repr(self):
        """Constructs the API resource of this TrainingOptions

        Returns:
            Dict[str, object]: TrainingOptions represented as an API resource.
        """
        return copy.deepcopy(self._properties)

    @property
    def max_iterations(self):
        """Union[int,None]:  max number of iterations in training.
        """
        return self._properties.get("maxIterations")

    @property
    def loss_type(self):
        """Union[str,None]:  type of loss function used during training run."
        """
        return self._properties.get("lossType")
    
    @property
    def learn_rate(self):
        """Union[double, None]: Learning rate in training.
        """
        return self._properties.get("learnRate")

    @property
    def l1_regularization(self):
        """Union[double, None]: L1 regularization coefficient.
        """
        return self._properties.get("l1Regularization")

    @property
    def l2_regularization(self):
        """Union[double, None]: L2 regularization coefficient.
        """
        return self._properties.get("l2Regularization")

    @property
    def min_relative_progress(self):
        """Union[double, None]: When early stop is true, stops training
        when accuracy improvement is less than min_relative_progress.
        """
        return self._properties.get("minRelativeProgress")

    @property
    def warm_start(self):
        """Union[bool, None]: whether to train model from last checkpoint.
        """
        return self._properties.get("warmStart")

    @property
    def early_stop(self):
        """Union[bool, None]: whether to stop early when loss doesn't improve
        significantly/any more compared to min_relative_progress.
        """
        return self._properties.get("earlyStop")

    @property
    def input_label_columns(self): 
        """List[str]: Name of input label columns in training data."""
        cols = self._properties.get("inputLabelColumns")
        if cols is None:
            return []
        return cols


class Model(object):
    """Models represent a machine learning model created within BigQuery.
    
    BigQuery models are not created directly via API, but via CREATE MODEL queries.

    See
    https://cloud.google.com/bigquery/docs/reference/rest/v2/models

    Args:
        model_ref (google.cloud.bigquery.model.ModelReference):
            A pointer to a model
    """

    _PROPERTY_TO_API_FIELD = {
        "friendly_name": "friendlyName",
    }

    def __init__(self, model_ref, schema=None):
        self._properties = {"modelReference": model_ref.to_api_repr(), "labels": {}}

    @property
    def project(self):
        """str: Project bound to the model."""
        return self._properties["modelReference"]["projectId"]

    @property
    def dataset_id(self):
        """str: ID of dataset containing the model."""
        return self._properties["modelReference"]["datasetId"]

    @property
    def model_id(self):
        """str: ID of the model."""
        return self._properties["modelReference"]["modelId"]

    reference = property(_reference_getter)

    @property
    def path(self):
        """str: URL path for the model's APIs."""
        return "/projects/%s/datasets/%s/models/%s" % (
            self.project,
            self.dataset_id,
            self.model_id,
        )

    @property
    def created(self):
        """Union[datetime.datetime, None]: Datetime at which the model was
        created (:data:`None` until set from the server).
        """
        creation_time = self._properties.get("creationTime")
        if creation_time is not None:
            # creation_time will be in milliseconds.
            return google.cloud._helpers._datetime_from_microseconds(
                1000.0 * float(creation_time)
            )

    @property
    def etag(self):
        """Union[str, None]: ETag for the model resource (:data:`None` until
        set from the server).
        """
        return self._properties.get("etag")

    @property
    def friendly_name(self):
        """Union[str, None]: Title of the model (defaults to :data:`None`).

        Raises:
            ValueError: For invalid value types.
        """
        return self._properties.get("friendlyName")

    @friendly_name.setter
    def friendly_name(self, value):
        if not isinstance(value, six.string_types) and value is not None:
            raise ValueError("Pass a string, or None")
        self._properties["friendlyName"] = value

    @property
    def modified(self):
        """Union[datetime.datetime, None]: Datetime at which the model was last
        modified (:data:`None` until set from the server).
        """
        modified_time = self._properties.get("lastModifiedTime")
        if modified_time is not None:
            # modified_time will be in milliseconds.
            return google.cloud._helpers._datetime_from_microseconds(
                1000.0 * float(modified_time)
            )

    @property
    def description(self):
        """Union[str, None]: Description of the model (defaults to
        :data:`None`).

        Raises:
            ValueError: For invalid value types.
        """
        return self._properties.get("description")

    @description.setter
    def description(self, value):
        if not isinstance(value, six.string_types) and value is not None:
            raise ValueError("Pass a string, or None")
        self._properties["description"] = value

    @property
    def expires(self):
        """Union[datetime.datetime, None]: Datetime at which the model will be
        deleted.

        Raises:
            ValueError: For invalid value types.
        """
        expiration_time = self._properties.get("expirationTime")
        if expiration_time is not None:
            # expiration_time will be in milliseconds.
            return google.cloud._helpers._datetime_from_microseconds(
                1000.0 * float(expiration_time)
            )

    @expires.setter
    def expires(self, value):
        if not isinstance(value, datetime.datetime) and value is not None:
            raise ValueError("Pass a datetime, or None")
        value_ms = google.cloud._helpers._millis_from_datetime(value)
        self._properties["expirationTime"] = _helpers._str_or_none(value_ms)

    @property
    def feature_columns(self):
        """List[google.cloud.bigquery.model.StandardSqlField]: Input feature columns
           used to train the model.
        """
        prop = self._properties.get("featureColumns")
        if not prop:
            return []
        else:
            return []
            #return _parse_sql_field_resources(prop)

    @property
    def label_columns(self):
        """List[google.cloud.bigquery.model.StandardSqlField]: Label columns
           used to train the model.  The output of the model will have a 'predicted_'
           prefix for these columns.
        """
        prop = self._properties.get("labelColumns")
        if not prop:
            return []
        else:
            return []
            #return _parse_sql_field_resources(prop)

    @property
    def labels(self):
        """Dict[str, str]: Labels for the model.

        This method always returns a dict. To change a models's labels,
        modify the dict, then call ``Client.update_model``. To delete a
        label, set its value to :data:`None` before updating.
        """
        return self._properties.setdefault("labels", {})

    @property
    def training_runs(self):
        """List[google.cloud.bigquery.model.TrainingRun]: Information for training run iterations
        ordered by ascending start times.
        """
        prop = self._properties.get("trainingRuns")
        if not prop:
            return []
        else:
            # hoist into a _parse_training_runs?
            runs = []
            for run in prop:
                training_run = TrainingRun.from_api_repr(run)
                runs.append(training_run)
            return runs

    @property
    def model_type(self):
        """Union[str, None]: The type of the model (:data:`None` until set from
        the server).

        Possible values include ``'LINEAR_REGRESSION'``, ``'LOGISTIC_REGRESSION'``, and ``'MODEL_TYPE_UNSPECIFIED'``.
        """
        return self._properties.get("modelType")

    @property
    def location(self):
        """Union[str, None]: Location in which the model is hosted

        Defaults to :data:`None`.
        """
        return self._properties.get("location")

    @classmethod
    def from_string(cls, full_model_id):
        """Construct a model from fully-qualified model ID.

        Args:
            full_model_id (str):
                A fully-qualified model ID in standard SQL format. Must
                included a project ID, dataset ID, and model ID, each
                separated by ``.``.

        Returns:
            Model: Model parsed from ``full_model_id``.

        Examples:
            >>> Model.from_string('my-project.mydataset.mymodel')
            Model(ModelRef...(D...('my-project', 'mydataset'), 'mymodel'))

        Raises:
            ValueError:
                If ``full_model_id`` is not a fully-qualified model ID in
                standard SQL format.
        """
        return cls(ModelReference.from_string(full_model_id))

    @classmethod
    def from_api_repr(cls, resource):
        """Factory: construct a model given its API representation

        Args:
            resource (Dict[str, object]):
                Model resource representation from the API
            dataset (google.cloud.bigquery.dataset.Dataset):
                The dataset containing the model.

        Returns:
            google.cloud.bigquery.model.Model: Model parsed from ``resource``.

        Raises:
            KeyError:
                If the ``resource`` lacks the key ``'modelReference'``, or if
                the ``dict`` stored within the key ``'modelReference'`` lacks
                the keys ``'modelId'``, ``'projectId'``, or ``'datasetId'``.
        """
        from google.cloud.bigquery import dataset

        if (
            "modelReference" not in resource
            or "modelId" not in resource["modelReference"]
        ):
            raise KeyError(
                "Resource lacks required identity information:"
                '["modelReference"]["modelId"]'
            )
        project_id = resource["modelReference"]["projectId"]
        model_id = resource["modelReference"]["modelId"]
        dataset_id = resource["modelReference"]["datasetId"]
        dataset_ref = dataset.DatasetReference(project_id, dataset_id)

        model = cls(dataset_ref.model(model_id))
        model._properties = resource

        return model

    def to_api_repr(self):
        """Constructs the API resource of this model

        Returns:
            Dict[str, object]: Model represented as an API resource
        """
        return copy.deepcopy(self._properties)

    def _build_resource(self, filter_fields):
        """Generate a resource for ``update``."""
        partial = {}
        for filter_field in filter_fields:
            api_field = self._PROPERTY_TO_API_FIELD.get(filter_field)
            if api_field is None and filter_field not in self._properties:
                raise ValueError("No Model property %s" % filter_field)
            elif api_field is not None:
                partial[api_field] = self._properties.get(api_field)
            else:
                # allows properties that are not defined in the library
                # and properties that have the same name as API resource key
                partial[filter_field] = self._properties[filter_field]

        return partial

    def __repr__(self):
        return "Model({})".format(repr(self.reference))


class ModelListItem(object):
    """A read-only model resource from a list operation.

    For performance reasons, the BigQuery API only includes some of the model
    properties when listing models. Specifically, only the following properties
    are included: etag, the id, model type, creation time, and last modified.
    time.

    Args:
        resource (Dict[str, object]):
            A model-like resource object from a model list response. A
            ``modelReference`` property is required.

    Raises:
        ValueError:
            If ``modelReference`` or one of its required members is missing
            from ``resource``.
    """

    def __init__(self, resource):
        if "modelReference" not in resource:
            raise ValueError("resource must contain a modelReference value")
        if "projectId" not in resource["modelReference"]:
            raise ValueError(
                "resource['modelReference'] must contain a projectId value"
            )
        if "datasetId" not in resource["modelReference"]:
            raise ValueError(
                "resource['modelReference'] must contain a datasetId value"
            )
        if "modelId" not in resource["modelReference"]:
            raise ValueError("resource['modelReference'] must contain a modelId value")

        self._properties = resource

    @property
    def project(self):
        """str: Project bound to the model."""
        return self._properties["modelReference"]["projectId"]

    @property
    def dataset_id(self):
        """str: ID of dataset containing the model."""
        return self._properties["modelReference"]["datasetId"]

    @property
    def model_id(self):
        """str: ID of the model."""
        return self._properties["modelReference"]["modelId"]

    reference = property(_reference_getter)

    @property
    def model_type(self):
        """Union[str, None]: The type of the model (:data:`None` until set from
        the server).

        Possible values include ``'LINEAR_REGRESSION'``, ``'LOGISTIC_REGRESSION'``, and ``'MODEL_TYPE_UNSPECIFIED'``.
        """
        return self._properties.get("model_type")

    @property
    def created(self):
        """Union[datetime.datetime, None]: Datetime at which the model was
        created (:data:`None` until set from the server).
        """
        creation_time = self._properties.get("creationTime")
        if creation_time is not None:
            # creation_time will be in milliseconds.
            return google.cloud._helpers._datetime_from_microseconds(
                1000.0 * float(creation_time)
            )

    @property
    def etag(self):
        """Union[str, None]: ETag for the model resource (:data:`None` until
        set from the server).
        """
        return self._properties.get("etag")

    @property
    def modified(self):
        """Union[datetime.datetime, None]: Datetime at which the model was last
        modified (:data:`None` until set from the server).
        """
        modified_time = self._properties.get("lastModifiedTime")
        if modified_time is not None:
            # modified_time will be in milliseconds.
            return google.cloud._helpers._datetime_from_microseconds(
                1000.0 * float(modified_time)
            )   

    @classmethod
    def from_string(cls, full_model_id):
        """Construct a model from fully-qualified model ID.

        Args:
            full_model_id (str):
                A fully-qualified model ID in standard SQL format. Must
                included a project ID, dataset ID, and model ID, each
                separated by ``.``.

        Returns:
            Model: Model parsed from ``full_model_id``.

        Examples:
            >>> Model.from_string('my-project.mydataset.mymodel')
            Model(ModelRef...(D...('my-project', 'mydataset'), 'mymodel'))

        Raises:
            ValueError:
                If ``full_model_id`` is not a fully-qualified model ID in
                standard SQL format.
        """
        return cls(
            {"modelReference": ModelReference.from_string(full_model_id).to_api_repr()}
        )

# pylint: enable=unused-argument
