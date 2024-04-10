from typing import List, Optional

from helm.benchmark.adaptation.common_adapter_specs import (
    format_instructions
)

from helm.benchmark.adaptation.adapter_spec import (
    ADAPT_GENERATION,
    AdapterSpec,
)

def get_bhasa_adapter_spec(
    instructions: str = "",
    input_noun: Optional[str] = None,
    newline_after_input_noun: bool = False,
    input_suffix: Optional[str] = None,
    output_noun: Optional[str] = None,
    newline_after_output_noun: bool = False,
    max_train_instances: int = 0,
    num_outputs: int = 1,
    max_tokens: int = 5,
    stop_sequences: Optional[List] = None,  # default value of `stop_sequences` is ["\n"]
    temperature: float = 0.0,
    multi_label: bool = False,
) -> AdapterSpec:
    """
    [instructions]

    [input_noun]: [input]
    [input_suffix]
    [output_noun]: [output]

    [input_noun]: [input]
    [input_suffix]
    [output_noun]:
    """

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

    if stop_sequences is None:
        stop_sequences = ["\n"]

    return AdapterSpec(
        method=ADAPT_GENERATION,
        instructions=instructions,
        input_prefix=format_prefix(input_noun, append_new_line=newline_after_input_noun),
        input_suffix=format_suffix(input_suffix),
        output_prefix=format_prefix(output_noun, append_new_line=newline_after_output_noun),
        max_train_instances=max_train_instances,
        num_outputs=num_outputs,
        max_tokens=max_tokens,
        temperature=temperature,
        stop_sequences=stop_sequences,
        multi_label=multi_label,
    )