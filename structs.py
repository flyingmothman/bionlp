from dataclasses import dataclass, field

@dataclass
class Annotation:
    begin_offset: int
    end_offset: int
    label_type: str
    extraction: str
    features: dict = field(default_factory=dict)


@dataclass
class AnnotationCollection:
    gold: list[Annotation]
    external: list[Annotation]


@dataclass
class Sample:
    text: str
    id: str
    annos: AnnotationCollection
    features: dict = field(default_factory=dict)
