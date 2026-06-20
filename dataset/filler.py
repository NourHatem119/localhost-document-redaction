"""PII-free boilerplate filler for padding documents to realistic length.

Real enterprise documents are mostly dense boilerplate — terms, disclaimers,
policy clauses — wrapped around a small amount of personal data. To make the
fixtures multi-page (so the chunker's overlapping-window path is actually
exercised) we pad each document with domain-appropriate filler paragraphs.

Hard rule: filler must contain NO PII-shaped text. No capitalised full names,
no emails, no phone numbers, no dates in numeric form, no card/IBAN/NI shapes.
Otherwise the detector would flag it and it would count as a false "miss"
against the gold labels. Paragraphs here are deliberately generic prose.
"""

from __future__ import annotations

from dataset.schema import DocType

# Roughly 450 words fill one A4 page at the generator's font/leading.
WORDS_PER_PAGE = 450


_COMMON = [
    "This document forms part of the organisation's official records and is "
    "retained in accordance with the applicable data retention schedule. The "
    "information set out below has been prepared for internal administrative "
    "purposes and should be treated as confidential at all times.",

    "Recipients are reminded that the contents must not be disclosed to any "
    "third party without prior written authorisation from the relevant "
    "department. Any unauthorised reproduction, distribution, or onward "
    "transmission of this material may constitute a breach of policy and of "
    "the relevant data protection obligations.",

    "Where the contents of this document are required for audit, compliance, "
    "or regulatory review, access shall be granted only to those individuals "
    "with a legitimate operational need. A record of all such access is "
    "maintained and reviewed periodically by the responsible team.",

    "Should any of the details recorded here prove to be inaccurate or "
    "out of date, the matter should be raised through the standard correction "
    "process so that the records can be updated promptly. The organisation is "
    "committed to maintaining accurate and proportionate records.",

    "This page has been left with explanatory notes to assist the reader in "
    "understanding the structure of the document. The sections that follow are "
    "presented in a consistent order to support straightforward review and to "
    "ensure that all required particulars are captured.",
]

_BY_TYPE = {
    DocType.EMPLOYMENT_LETTER: [
        "Your continued employment is subject to the terms and conditions set "
        "out in the staff handbook, which is available on the internal portal. "
        "Those terms cover working hours, leave entitlement, notice periods, "
        "and the conduct expected of all members of staff.",

        "The organisation operates a probationary period during which "
        "performance and suitability for the role are reviewed. Objectives will "
        "be agreed with your line manager and progress discussed at regular "
        "intervals throughout the period.",

        "Pension arrangements are provided through the workplace scheme in line "
        "with automatic enrolment requirements. Further information about "
        "contribution rates and investment options can be obtained from the "
        "people team on request.",
    ],
    DocType.HR_ONBOARDING: [
        "All new starters are required to complete the mandatory induction "
        "modules within their first two weeks. These cover health and safety, "
        "information security, and the code of conduct that applies across the "
        "organisation.",

        "Right-to-work checks are carried out for every new starter in "
        "accordance with statutory requirements. The verifying officer confirms "
        "that the documents presented are genuine, current, and belong to the "
        "individual presenting them.",

        "Equipment and system access will be provisioned once the onboarding "
        "checklist has been completed and approved. Access is granted on a "
        "least-privilege basis and reviewed at regular intervals thereafter.",
    ],
    DocType.BANK_STATEMENT: [
        "Please review this statement carefully and report any transaction you "
        "do not recognise as soon as possible. Prompt reporting helps to "
        "protect your account and allows any necessary investigation to begin "
        "without delay.",

        "Interest, where applicable, is calculated on the daily cleared balance "
        "and applied in accordance with the published rates for the account. "
        "Charges for additional services are set out in the current tariff "
        "guide.",

        "This statement is provided for your records. It is recommended that "
        "statements are stored securely and disposed of carefully when no "
        "longer required, given the sensitive nature of the information they "
        "contain.",

        "Your account is covered by the relevant deposit protection scheme up "
        "to the published limit. Details of the scheme and the protection it "
        "provides are available on request or through the published guidance.",
    ],
    DocType.MEDICAL_RECORD: [
        "This record is maintained as part of the patient's clinical history "
        "and is shared only with those directly involved in the patient's care "
        "or as otherwise permitted under the applicable confidentiality "
        "framework.",

        "Clinical entries should be read in the context of the full record. "
        "Where further clarification is required, the responsible clinician "
        "should be consulted before any decision is made on the basis of the "
        "information recorded here.",

        "Appointments and follow-up arrangements are subject to clinical "
        "prioritisation and may be adjusted in light of changing needs. "
        "Patients are advised to attend scheduled reviews unless told "
        "otherwise.",
    ],
    DocType.INVOICE: [
        "Payment is due within the terms stated on this invoice. Where payment "
        "is not received by the due date, the account may be subject to a "
        "reminder process and, where applicable, charges in accordance with the "
        "agreed terms of business.",

        "All amounts are stated in pounds sterling unless otherwise indicated. "
        "Queries relating to this invoice should be raised through the accounts "
        "function quoting the reference shown, so that they can be resolved "
        "promptly.",

        "Goods and services supplied remain subject to the standard terms and "
        "conditions of sale. A copy of those terms is available on request and "
        "governs the relationship between the parties in respect of this "
        "supply.",
    ],
}


def filler_paragraphs(doc_type: DocType) -> list[str]:
    """Ordered pool of PII-free paragraphs appropriate to the document type."""
    return _BY_TYPE.get(doc_type, []) + _COMMON


def pad_to_words(builder, doc_type: DocType, target_words: int) -> None:
    """Append filler paragraphs to `builder` until it reaches ~target_words.

    Paragraphs are appended through the builder's plain `add()`, so no PII
    spans are recorded and document offsets stay correct by construction.
    """
    pool = filler_paragraphs(doc_type)
    if not pool:
        return
    i = 0
    while len(builder.text().split()) < target_words:
        builder.add("\n\n").add(pool[i % len(pool)])
        i += 1
