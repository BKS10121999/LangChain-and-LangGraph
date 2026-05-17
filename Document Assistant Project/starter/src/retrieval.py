from dataclasses import dataclass
import math
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from schemas import DocumentChunk


@dataclass
class Document:
    """Represents a source document in the local document store."""

    doc_id: str
    title: str
    content: str
    doc_type: str
    metadata: Dict[str, Any]


class SimulatedRetriever:
    """
    Metadata-aware lexical retriever used by the assistant and tools.

    This project does not ship a real vector database, so this class implements
    the same production retrieval stages in a lightweight form: query analysis,
    metadata filtering, chunk scoring, and MMR-style diversification. The
    public methods preserve the original API while adding a stronger
    `retrieve(...)` entry point for graph nodes.
    """

    STOPWORDS = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "in", "is", "it", "of", "on", "or", "the", "to", "was", "what",
        "which", "who", "with", "all", "me", "show", "give", "please",
    }

    DOC_TYPE_TERMS = {
        "invoice": {
            "invoice", "invoices", "bill", "billing", "payment", "due",
            "revenue", "financial", "finance", "quarterly", "annual",
        },
        "contract": {"contract", "contracts", "agreement", "service agreement"},
        "claim": {
            "claim", "claims", "insurance", "patient", "diagnosis", "medical",
            "claimant", "policy", "hospital", "medication", "healthcare", "health",
        },
        "report": {"report", "reports", "annual report", "quarterly report"},
    }

    CATEGORY_BY_TYPE = {
        "invoice": "financial",
        "contract": "legal",
        "claim": "medical",
        "report": "financial_report",
    }

    def __init__(self):
        self.documents: Dict[str, Document] = {}
        self._chunks: List[DocumentChunk] = []
        self._load_sample_documents()

    def _load_sample_documents(self):
        """Load sample documents with normalized metadata for filtering."""
        sample_docs = [
            Document(
                doc_id="INV-001",
                title="Invoice #12345",
                content="""
                Invoice #12345
                Date: 2024-01-15
                Client: Acme Corporation

                Services Rendered:
                - Consulting Services: $5,000
                - Software Development: $12,500
                - Support & Maintenance: $2,500

                Subtotal: $20,000
                Tax (10%): $2,000

                Payment Terms: Net 30 days
                """,
                doc_type="invoice",
                metadata={"customer_name": "Acme Corporation", "client": "Acme Corporation", "date": "2024-01-15"},
            ),
            Document(
                doc_id="CON-001",
                title="Service Agreement",
                content="""
                SERVICE AGREEMENT

                This Service Agreement is entered into on January 1, 2024, between:
                - Provider: DocDacity Solutions Inc.
                - Client: Healthcare Partners LLC

                Services:
                1. Document Processing Platform Access
                2. 24/7 Technical Support
                3. Monthly Data Analytics Reports
                4. Compliance Monitoring

                Duration: 12 months
                Monthly Fee: $15,000
                Total Contract Value: $180,000

                Termination: Either party may terminate with 60 days written notice.
                """,
                doc_type="contract",
                metadata={
                    "value": 180000,
                    "duration_months": 12,
                    "customer_name": "Healthcare Partners LLC",
                    "client": "Healthcare Partners LLC",
                },
            ),
            Document(
                doc_id="CLM-001",
                title="Insurance Claim #78901",
                content="""
                INSURANCE CLAIM FORM
                Claim Number: 78901
                Date of Incident: 2024-02-10
                Policy Number: POL-456789

                Claimant: John Doe
                Type of Claim: Medical Expense Reimbursement

                Expenses:
                - Hospital Visit: $1,200
                - Diagnostic Tests: $800
                - Medication: $150
                - Follow-up Consultation: $300

                Total Claim Amount: $2,450

                Status: Under Review
                """,
                doc_type="claim",
                metadata={
                    "amount": 2450,
                    "status": "Under Review",
                    "claimant": "John Doe",
                    "customer_name": "John Doe",
                    "patient_name": "John Doe",
                },
            ),
            Document(
                doc_id="INV-002",
                title="Invoice #12346",
                content="""
                Invoice #12346
                Date: 2024-02-20
                Client: TechStart Inc.

                Products:
                - Enterprise License (Annual): $50,000
                - Implementation Services: $15,000
                - Training Package: $5,000

                Subtotal: $70,000
                Discount (10%): -$7,000
                Tax (10%): $6,300
                Total Due: $69,300

                Payment Terms: Net 45 days
                """,
                doc_type="invoice",
                metadata={"total": 69300, "customer_name": "TechStart Inc.", "client": "TechStart Inc.", "date": "2024-02-20"},
            ),
            Document(
                doc_id="INV-003",
                title="Invoice #12347",
                content="""
                Invoice #12347
                Date: 2024-03-01
                Client: Global Corp

                Services:
                - Annual Subscription: $120,000
                - Premium Support: $30,000
                - Custom Development: $45,000

                Subtotal: $195,000
                Tax (10%): $19,500
                Total Due: $214,500

                Payment Terms: Net 60 days
                """,
                doc_type="invoice",
                metadata={"total": 214500, "customer_name": "Global Corp", "client": "Global Corp", "date": "2024-03-01"},
            ),
        ]

        for doc in sample_docs:
            self.add_document(doc)

    def load_documents(self, documents: Optional[List[Document]] = None) -> None:
        """Replace the collection and rebuild the retrieval index."""
        self.documents = {}
        if documents is None:
            self._load_sample_documents()
            return

        for doc in documents:
            self.add_document(doc)

    def add_document(self, document: Document):
        """Add a document and rebuild chunks so retrieval stays consistent."""
        metadata = self._normalize_metadata(document)
        self.documents[document.doc_id] = Document(
            doc_id=document.doc_id,
            title=document.title,
            content=document.content,
            doc_type=document.doc_type.lower(),
            metadata=metadata,
        )
        self._rebuild_index()

    def analyze_query(self, query: str, intent_type: str = "qa") -> Dict[str, Any]:
        """Infer metadata filters, top_k, and ambiguity from the user query."""
        query_lower = query.lower().strip()
        doc_ids = self.extract_document_ids(query)
        doc_types = self.infer_doc_types(query, intent_type)
        category = None
        if "report" in doc_types:
            category = self.CATEGORY_BY_TYPE.get("report")
        elif len(doc_types) == 1:
            category = self.CATEGORY_BY_TYPE.get(doc_types[0])

        top_k = 5 if intent_type in {"summarization", "calculation"} else 3
        if doc_ids:
            top_k = max(top_k, len(doc_ids))
        if intent_type == "calculation" and "invoice" in doc_types and any(term in query_lower for term in ["all", "total", "sum"]):
            top_k = 20

        ambiguous = self.is_ambiguous(query, intent_type, doc_ids, doc_types)
        return {
            "rewritten_query": self.rewrite_query(query, intent_type, doc_types),
            "doc_ids": doc_ids,
            "doc_types": doc_types,
            "category": category,
            "top_k": top_k,
            "ambiguous": ambiguous,
            "reason": self._plan_reason(intent_type, doc_ids, doc_types, ambiguous),
        }

    def retrieve(
        self,
        query: str,
        intent_type: str = "qa",
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        use_mmr: bool = True,
    ) -> Tuple[List[DocumentChunk], Dict[str, Any]]:
        """Run metadata-aware retrieval and return chunks plus diagnostics."""
        plan = self.analyze_query(query, intent_type)
        filters = {**plan, **(filters or {})}
        requested_top_k = top_k or int(filters.get("top_k") or 3)

        candidates = self._filter_chunks(
            doc_ids=filters.get("doc_ids") or [],
            doc_types=filters.get("doc_types") or [],
            category=filters.get("category"),
        )

        query_text = filters.get("rewritten_query") or query
        scored = [
            self._score_chunk(query_text, chunk, filters.get("doc_types") or [])
            for chunk in candidates
        ]
        scored = [chunk for chunk in scored if chunk.relevance_score > 0]
        scored.sort(key=lambda c: c.relevance_score, reverse=True)

        if use_mmr:
            results = self._mmr(scored, requested_top_k)
        else:
            results = scored[:requested_top_k]

        diagnostics = {
            "query": query,
            "rewritten_query": query_text,
            "intent_type": intent_type,
            "filters": {
                "doc_ids": filters.get("doc_ids") or [],
                "doc_types": filters.get("doc_types") or [],
                "category": filters.get("category"),
            },
            "candidate_count": len(candidates),
            "returned_count": len(results),
            "top_k": requested_top_k,
            "results": [
                {
                    "doc_id": chunk.doc_id,
                    "doc_type": chunk.metadata.get("doc_type"),
                    "score": round(chunk.relevance_score, 4),
                    "title": chunk.metadata.get("title"),
                }
                for chunk in results
            ],
            "plan_reason": filters.get("reason"),
        }
        return results, diagnostics

    def retrieve_documents(self, query: str, top_k: int = 3) -> List[Dict[str, str]]:
        chunks, _ = self.retrieve(query, top_k=top_k)
        return [{"doc_id": c.doc_id, "relevant_text": c.content.strip()[:1000]} for c in chunks]

    def retrieve_all(self) -> List[DocumentChunk]:
        return [self._document_to_chunk(doc) for doc in self.documents.values()]

    def retrieve_by_keyword(
        self,
        query: str,
        top_k: int = 3,
        doc_type: Optional[str] = None,
        category: Optional[str] = None,
    ) -> List[DocumentChunk]:
        filters: Dict[str, Any] = {"top_k": top_k}
        if doc_type:
            filters["doc_types"] = [doc_type.lower()]
        if category:
            filters["category"] = category
        results, _ = self.retrieve(query, top_k=top_k, filters=filters)
        return results

    def retrieve_by_type(self, doc_type: str) -> List[DocumentChunk]:
        return [
            self._document_to_chunk(doc)
            for doc in self.documents.values()
            if doc.doc_type.lower() == doc_type.lower()
        ]

    def retrieve_by_amount_range(
        self,
        min_amount: Optional[float] = None,
        max_amount: Optional[float] = None,
        doc_type: Optional[str] = None,
    ) -> List[DocumentChunk]:
        results = []
        for doc in self.documents.values():
            if doc_type and doc.doc_type != doc_type.lower():
                continue
            amount = self._get_document_amount(doc)
            if amount is None:
                continue
            if min_amount is not None and amount < min_amount:
                continue
            if max_amount is not None and amount > max_amount:
                continue
            chunk = self._document_to_chunk(doc)
            chunk.relevance_score = 1.0
            results.append(chunk)

        results.sort(key=lambda x: self._get_document_amount_from_chunk(x), reverse=True)
        return results

    def retrieve_by_exact_amount(self, amount: float, tolerance: float = 0.01) -> List[DocumentChunk]:
        return [
            self._document_to_chunk(doc)
            for doc in self.documents.values()
            if (doc_amount := self._get_document_amount(doc)) is not None
            and abs(doc_amount - amount) <= tolerance
        ]

    def retrieve_by_approximate_amount(self, amount: float, percentage: float = 10.0) -> List[DocumentChunk]:
        tolerance = amount * (percentage / 100)
        return self.retrieve_by_amount_range(amount - tolerance, amount + tolerance)

    def retrieve_by_amount(
        self,
        query: str,
        comparison_type: Optional[str] = None,
        amount: Optional[float] = None,
        min_amount: Optional[float] = None,
        max_amount: Optional[float] = None,
    ) -> List[DocumentChunk]:
        query_types = self.infer_doc_types(query, "calculation")
        doc_type = query_types[0] if len(query_types) == 1 else None
        if comparison_type:
            if comparison_type in ["greater", "over", "above", "more than"]:
                return self.retrieve_by_amount_range(min_amount=amount, doc_type=doc_type)
            if comparison_type in ["less", "under", "below", "less than"]:
                return self.retrieve_by_amount_range(max_amount=amount, doc_type=doc_type)
            if comparison_type in ["exact", "exactly", "equal", "equals"] and amount is not None:
                return self.retrieve_by_exact_amount(amount)
            if comparison_type in ["approximate", "around", "about", "roughly"] and amount is not None:
                return self.retrieve_by_approximate_amount(amount)
            if comparison_type in ["between", "range"]:
                return self.retrieve_by_amount_range(min_amount=min_amount, max_amount=max_amount, doc_type=doc_type)
        return self._parse_and_retrieve_by_amount(query)

    def get_document_by_id(self, doc_id: str) -> Optional[DocumentChunk]:
        doc = self.documents.get(doc_id.upper())
        return self._document_to_chunk(doc) if doc else None

    def get_statistics(self) -> Dict[str, Any]:
        total_docs = len(self.documents)
        docs_with_amounts = 0
        total_amount = 0.0
        amounts = []
        doc_types: Dict[str, int] = {}

        for doc in self.documents.values():
            doc_types[doc.doc_type] = doc_types.get(doc.doc_type, 0) + 1
            amount = self._get_document_amount(doc)
            if amount is not None:
                docs_with_amounts += 1
                total_amount += amount
                amounts.append(amount)

        stats = {
            "total_documents": total_docs,
            "documents_with_amounts": docs_with_amounts,
            "total_amount": total_amount,
            "average_amount": total_amount / docs_with_amounts if docs_with_amounts else 0,
            "document_types": doc_types,
        }
        if amounts:
            stats["min_amount"] = min(amounts)
            stats["max_amount"] = max(amounts)
        return stats

    def extract_financial_values(self, chunks: Sequence[DocumentChunk]) -> List[Dict[str, Any]]:
        """Extract one reliable amount per document for calculation workflows."""
        values = []
        for chunk in self._unique_document_chunks(chunks):
            doc = self.documents.get(chunk.doc_id)
            if not doc:
                continue
            amount, label = self._extract_best_amount(doc)
            if amount is None:
                continue
            values.append({
                "doc_id": doc.doc_id,
                "doc_type": doc.doc_type,
                "label": label,
                "amount": amount,
                "title": doc.title,
            })
        return values

    def infer_doc_types(self, query: str, intent_type: str = "qa") -> List[str]:
        query_lower = query.lower()
        inferred = set()
        for doc_type, terms in self.DOC_TYPE_TERMS.items():
            if any(term in query_lower for term in terms):
                inferred.add(doc_type)

        if intent_type == "calculation" and not inferred:
            if any(term in query_lower for term in ["amount", "total", "sum", "balance", "due", "revenue"]):
                inferred.add("invoice")

        # A "financial report" is a report request; fail closed if no report is
        # present instead of summarizing invoices. Generic revenue/financial
        # calculation questions, however, can use invoice-like finance docs.
        if "report" in inferred:
            inferred = {"report"}
        elif "claim" in inferred and any(term in query_lower for term in ["healthcare", "medical", "patient", "diagnosis"]):
            inferred = {"claim"}

        # Keep requested-but-missing types such as "report" in the plan. That
        # lets summarization fail closed instead of falling back to unrelated
        # invoices or contracts.
        return sorted(inferred)

    def extract_document_ids(self, query: str) -> List[str]:
        ids = [match.upper() for match in re.findall(r"\b(?:INV|CON|CLM)-\d+\b", query, flags=re.IGNORECASE)]
        return list(dict.fromkeys(ids))

    def is_ambiguous(self, query: str, intent_type: str, doc_ids: Sequence[str], doc_types: Sequence[str]) -> bool:
        query_lower = query.lower().strip()
        vague_amount = re.fullmatch(r"(what|which)?\s*(is|was)?\s*(the\s*)?(amount|total|balance|value)\??", query_lower)
        if vague_amount and not doc_ids and not doc_types:
            return True
        if query_lower in {"summarize it", "summarise it", "what about it"} and not doc_ids and not doc_types:
            return True
        return False

    def rewrite_query(self, query: str, intent_type: str, doc_types: Sequence[str]) -> str:
        additions = []
        if "claim" in doc_types:
            additions.extend(["medical", "patient", "claim", "insurance"])
        if "invoice" in doc_types:
            additions.extend(["invoice", "total", "due", "client"])
        if "contract" in doc_types:
            additions.extend(["contract", "agreement", "client", "value"])
        if intent_type == "summarization":
            additions.append("summary key points")
        return " ".join([query, *additions]).strip()

    def _normalize_metadata(self, document: Document) -> Dict[str, Any]:
        metadata = dict(document.metadata or {})
        metadata["document_id"] = document.doc_id
        metadata["doc_type"] = document.doc_type.lower()
        metadata["title"] = document.title
        metadata.setdefault("category", self.CATEGORY_BY_TYPE.get(document.doc_type.lower(), "general"))
        if "customer_name" not in metadata and "client" in metadata:
            metadata["customer_name"] = metadata["client"]
        return metadata

    def _rebuild_index(self) -> None:
        self._chunks = []
        for doc in self.documents.values():
            for index, text in enumerate(self._split_document(doc.content)):
                metadata = dict(doc.metadata)
                metadata["chunk_index"] = index
                self._chunks.append(DocumentChunk(
                    doc_id=doc.doc_id,
                    content=text,
                    metadata=metadata,
                    relevance_score=0.0,
                ))

    def _split_document(self, content: str, max_chars: int = 900, overlap: int = 120) -> List[str]:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", content) if p.strip()]
        chunks = []
        current = ""
        for paragraph in paragraphs:
            if len(current) + len(paragraph) + 2 <= max_chars:
                current = f"{current}\n\n{paragraph}".strip()
            else:
                if current:
                    chunks.append(current)
                current = paragraph
        if current:
            chunks.append(current)

        expanded = []
        for chunk in chunks:
            if len(chunk) <= max_chars:
                expanded.append(chunk)
                continue
            start = 0
            while start < len(chunk):
                expanded.append(chunk[start:start + max_chars])
                start += max_chars - overlap
        return expanded or [content.strip()]

    def _filter_chunks(self, doc_ids: Sequence[str], doc_types: Sequence[str], category: Optional[str]) -> List[DocumentChunk]:
        doc_id_set = {doc_id.upper() for doc_id in doc_ids}
        type_set = {doc_type.lower() for doc_type in doc_types}
        chunks = []
        for chunk in self._chunks:
            if doc_id_set and chunk.doc_id not in doc_id_set:
                continue
            if type_set and chunk.metadata.get("doc_type") not in type_set:
                continue
            if category and chunk.metadata.get("category") != category:
                continue
            chunks.append(chunk.model_copy(deep=True))
        return chunks

    def _score_chunk(self, query: str, chunk: DocumentChunk, requested_types: Sequence[str]) -> DocumentChunk:
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return chunk

        title_tokens = self._tokenize(str(chunk.metadata.get("title", "")))
        content_tokens = self._tokenize(chunk.content)
        metadata_tokens = self._tokenize(" ".join(str(v) for v in chunk.metadata.values()))

        score = 0.0
        for token in query_tokens:
            if token in title_tokens:
                score += 3.0
            score += content_tokens.count(token) * 1.0
            if token in metadata_tokens:
                score += 2.0

        doc_type = chunk.metadata.get("doc_type")
        if doc_type in requested_types:
            score += 5.0
        if chunk.doc_id.lower() in query.lower():
            score += 10.0

        chunk.relevance_score = score / max(math.sqrt(len(content_tokens) or 1), 1.0)
        return chunk

    def _mmr(self, chunks: Sequence[DocumentChunk], top_k: int, lambda_mult: float = 0.75) -> List[DocumentChunk]:
        selected: List[DocumentChunk] = []
        remaining = list(chunks)
        while remaining and len(selected) < top_k:
            if not selected:
                selected.append(remaining.pop(0))
                continue
            best_index = 0
            best_score = float("-inf")
            for index, chunk in enumerate(remaining):
                diversity_penalty = max(self._chunk_similarity(chunk, chosen) for chosen in selected)
                mmr_score = lambda_mult * chunk.relevance_score - (1 - lambda_mult) * diversity_penalty
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_index = index
            selected.append(remaining.pop(best_index))
        return selected

    def _chunk_similarity(self, left: DocumentChunk, right: DocumentChunk) -> float:
        left_tokens = set(self._tokenize(left.content))
        right_tokens = set(self._tokenize(right.content))
        if not left_tokens or not right_tokens:
            return 0.0
        return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)

    def _parse_and_retrieve_by_amount(self, query: str) -> List[DocumentChunk]:
        query_lower = query.lower()
        amounts = [float(m.replace(",", "")) for m in re.findall(r"\$?(\d+(?:,\d{3})*(?:\.\d{2})?)", query)]
        doc_type = self.infer_doc_types(query, "calculation")
        type_filter = doc_type[0] if len(doc_type) == 1 else None

        if any(word in query_lower for word in ["over", "above", "more than", "greater than", ">"]) and amounts:
            return self.retrieve_by_amount_range(min_amount=amounts[0], doc_type=type_filter)
        if any(word in query_lower for word in ["under", "below", "less than", "<"]) and amounts:
            return self.retrieve_by_amount_range(max_amount=amounts[0], doc_type=type_filter)
        if any(word in query_lower for word in ["between", "range", "from"]) and len(amounts) >= 2:
            return self.retrieve_by_amount_range(min(amounts), max(amounts), doc_type=type_filter)
        return self.retrieve_by_keyword(query, doc_type=type_filter)

    def _document_to_chunk(self, doc: Document) -> DocumentChunk:
        return DocumentChunk(
            doc_id=doc.doc_id,
            content=doc.content,
            metadata=dict(doc.metadata),
            relevance_score=1.0,
        )

    def _unique_document_chunks(self, chunks: Sequence[DocumentChunk]) -> List[DocumentChunk]:
        seen = set()
        unique = []
        for chunk in chunks:
            if chunk.doc_id in seen:
                continue
            seen.add(chunk.doc_id)
            unique.append(self._document_to_chunk(self.documents[chunk.doc_id]))
        return unique

    def _get_document_amount(self, doc: Document) -> Optional[float]:
        amount, _ = self._extract_best_amount(doc)
        return amount

    def _extract_best_amount(self, doc: Document) -> Tuple[Optional[float], Optional[str]]:
        for field in ["total", "amount", "value", "total_amount", "total_value"]:
            if field in doc.metadata and doc.metadata[field] is not None:
                try:
                    return float(doc.metadata[field]), field
                except (TypeError, ValueError):
                    pass

        patterns = [
            ("Total Due", r"Total Due:\s*\$?([\d,]+(?:\.\d+)?)"),
            ("Total Claim Amount", r"Total Claim Amount:\s*\$?([\d,]+(?:\.\d+)?)"),
            ("Total Contract Value", r"Total Contract Value:\s*\$?([\d,]+(?:\.\d+)?)"),
            ("Subtotal", r"Subtotal:\s*\$?([\d,]+(?:\.\d+)?)"),
            ("Monthly Fee", r"Monthly Fee:\s*\$?([\d,]+(?:\.\d+)?)"),
        ]
        for label, pattern in patterns:
            match = re.search(pattern, doc.content, flags=re.IGNORECASE)
            if match:
                return float(match.group(1).replace(",", "")), label
        return None, None

    def _get_document_amount_from_chunk(self, chunk: DocumentChunk) -> float:
        doc = self.documents.get(chunk.doc_id)
        if not doc:
            return 0.0
        return self._get_document_amount(doc) or 0.0

    def _tokenize(self, text: str) -> List[str]:
        return [
            token
            for token in re.findall(r"[a-zA-Z0-9]+", text.lower())
            if token not in self.STOPWORDS and len(token) > 1
        ]

    def _plan_reason(self, intent_type: str, doc_ids: Sequence[str], doc_types: Sequence[str], ambiguous: bool) -> str:
        if ambiguous:
            return "Query is underspecified and needs clarification."
        if doc_ids:
            return f"Explicit document ID filter: {', '.join(doc_ids)}."
        if doc_types:
            return f"Intent {intent_type} mapped query to document type filter: {', '.join(doc_types)}."
        return "No metadata filter inferred; using scored lexical retrieval."
