from typing import List, Optional

from helm.benchmark.adaptation.common_adapter_specs import (
    format_instructions
)

from helm.benchmark.adaptation.adapter_spec import (
    ADAPT_GENERATION,
    ADAPT_MULTIPLE_CHOICE_JOINT,
    ADAPT_MULTIPLE_CHOICE_SEPARATE_ORIGINAL,
    AdapterSpec,
)

def format_prefix(noun: Optional[str], append_new_line: bool) -> str:
        """
        When `append_new_line` is False:
            [input_noun]: [input]

        When `append_new_line` is True:
            [input_noun]:
            [input]
        """
        prefix: str = f"{noun}:" if noun is not None else ""
        if len(prefix) > 0:
            prefix += "\n" if append_new_line else " "
        return prefix
    
def format_suffix(suffix: Optional[str]) -> str:
    suffix: str = f"\n{suffix}" if suffix is not None else ""
    return suffix + "\n"

def get_bhasa_adapter_spec(
    method: str = ADAPT_GENERATION,
    instructions: str = "",
    input_noun: Optional[str] = None,
    newline_after_input_noun: bool = False,
    input_prefix: Optional[str] = None,
    input_suffix: Optional[str] = None,
    output_noun: Optional[str] = None,
    newline_after_output_noun: bool = False,
    max_train_instances: int = 0,
    num_outputs: int = 1,
    max_tokens: int = 5,
    stop_sequences: Optional[List] = None,
    temperature: float = 0.0,
    multi_label: bool = False,
) -> AdapterSpec:
    """
    [instructions]

    [input_prefix]
    [input]
    [input_suffix]
    [output_noun]:
    [output]

    [input_prefix]
    [input]
    [input_suffix]
    [output_noun]:
    """

    if stop_sequences is None:
        stop_sequences = ["\n"]

    return AdapterSpec(
        method=ADAPT_GENERATION,
        instructions=instructions,
        input_prefix=input_prefix if input_prefix else format_prefix(input_noun, append_new_line=newline_after_input_noun),
        input_suffix=format_suffix(input_suffix),
        output_prefix=format_prefix(output_noun, append_new_line=newline_after_output_noun),
        max_train_instances=max_train_instances,
        num_outputs=num_outputs,
        max_tokens=max_tokens,
        temperature=temperature,
        stop_sequences=stop_sequences,
        multi_label=multi_label,
    )

def get_bhasa_multiple_choice_joint_adapter_spec(
    method: str = ADAPT_MULTIPLE_CHOICE_SEPARATE_ORIGINAL,
    instructions: str = "",
    input_noun: Optional[str] = None,
    newline_after_input_noun: bool = False,
    input_prefix: Optional[str] = None,
    input_suffix: Optional[str] = None,
    output_noun: Optional[str] = None,
    newline_after_output_noun: bool = False,
    num_outputs: int = 5,
    max_train_instances: int = 5,
    max_tokens: int = 5,
    sample_train: bool = True,
    stop_sequences: Optional[List] = None,
    temperature: float = 0.0,
    **kwargs,
) -> AdapterSpec:
    """
    [input_prefix]
    [input]
    [input_suffix]
    [output_noun]:
    [output]

    [input_prefix]
    [input]
    [input_suffix]
    [output_noun]:
    """

    return AdapterSpec(
        method=method,
        instructions=format_instructions(instructions),
        input_prefix=input_prefix if input_prefix else format_prefix(input_noun, append_new_line=newline_after_input_noun),
        input_suffix=format_suffix(input_suffix),
        output_prefix=format_prefix(output_noun, append_new_line=newline_after_output_noun),
        max_train_instances=max_train_instances,
        num_outputs=num_outputs,
        max_tokens=max_tokens,
        temperature=temperature,
        stop_sequences=stop_sequences,
        sample_train=sample_train,
        **kwargs,
    )