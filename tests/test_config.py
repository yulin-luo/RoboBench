from copy import deepcopy

import pytest
from pydantic import ValidationError

from robobench.config import BenchmarkConfig


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("paths", "middle_file_dir", "data/extra"),
        ("paths", "old_prefix", "/old/root"),
    ],
)
def test_removed_path_fields_are_rejected(config_data, section, field, value):
    invalid = deepcopy(config_data)
    invalid[section][field] = value

    with pytest.raises(ValidationError):
        BenchmarkConfig.model_validate(invalid)


def test_removed_dimension_prompt_field_is_rejected(config_data):
    invalid = deepcopy(config_data)
    invalid["dimensions"]["perception_reasoning"]["system_prompt_key"] = "perception"

    with pytest.raises(ValidationError):
        BenchmarkConfig.model_validate(invalid)
