from typing import Any, Iterable, List, Tuple, TypeAlias, TypeVar, TypedDict

QueryIdsType: TypeAlias = Iterable[int]
QueryIdsBatchType: TypeAlias = Iterable[QueryIdsType]
CnfType: TypeAlias = Iterable[Iterable[QueryIdsType]]

class ErrorResponse(TypedDict):
    error: str

T = TypeVar("T")
InfiniGramEngineResponse: TypeAlias = ErrorResponse | T


class FindResponse(TypedDict):
    cnt: int
    segment_by_shard: List[Tuple[int, int]]

class FindCnfResponse(TypedDict):
    cnt: int
    approx: bool
    ptrs_by_shard: List[List[int]]

class CountResponse(TypedDict):
    count: int
    approx: bool

class ProbResponse(TypedDict):
    prompt_cnt: int
    cont_cnt: int
    prob: float

ProbSequenceResponse: TypeAlias = List[ProbResponse]
ProbBatchedSequenceResponse: TypeAlias = List[ProbSequenceResponse]

class DistTokenResult(TypedDict):
    cont_cnt: int
    prob: float

class NtdResponse(TypedDict):
    prompt_cnt: int
    result_by_token_id: dict[int, DistTokenResult]
    approx: bool

NtdSequenceResponse: TypeAlias = List[NtdResponse]
NtdBatchedSequenceResponse: TypeAlias = List[NtdSequenceResponse]

class InfGramProbResponse(ProbResponse, TypedDict):
    suffix_len: int

InfGramProbSequenceResponse: TypeAlias = List[InfGramProbResponse]
InfGramProbBatchedSequenceResponse: TypeAlias = List[InfGramProbSequenceResponse]

class InfGramNtdResponse(NtdResponse, TypedDict):
    prompt_cnt: int
    result_by_token_id: dict[int, DistTokenResult]
    approx: bool
    suffix_len: int

InfGramNtdSequenceResponse: TypeAlias = List[InfGramNtdResponse]
InfGramNtdBatchedSequenceResponse: TypeAlias = List[InfGramNtdSequenceResponse]

class DocResult(TypedDict):
    doc_ix: int
    doc_len: int
    disp_len: int
    needle_offset: int
    metadata: str
    token_ids: List[int]
    blocked: bool = False

class SearchDocsResponse(TypedDict):
    cnt: int
    approx: bool
    idxs: List[int]
    documents: List[DocResult]

class CreativityResponse(TypedDict):
    rs: List[int]

class AttributionDoc(TypedDict):
    s: int
    ptr: int

class AttributionSpan(TypedDict):
    l: int
    r: int
    length: int
    count: int
    unigram_logprob_sum: float
    docs: List[AttributionDoc]

class AttributionResponse(TypedDict):
    spans: List[AttributionSpan]

class GetDocsByPtrsRequestWithTakedown(TypedDict):
    docs: List[AttributionDoc]
    span_ids: List[int]
    needle_len: int
    max_ctx_len: int
