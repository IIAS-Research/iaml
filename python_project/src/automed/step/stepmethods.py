@classmethod
def available_steps(cls) -> list:
    return [step for step in (super().available_steps() + cls.__include_step()) if step not in cls.__exclude_step()]
    