"""Approved BANKING77 identity; labels retain upstream spelling exactly."""

from dataclasses import dataclass

REVISION = "57ec275d8078af65b7731c2a98be812d844a6d6b"
SOURCE = "https://github.com/PolyAI-LDN/task-specific-datasets"
METADATA = "https://huggingface.co/datasets/PolyAI/banking77/raw/main/dataset_infos.json"
CATEGORIES = frozenset(
    [
        "card_arrival",
        "card_linking",
        "exchange_rate",
        "card_payment_wrong_exchange_rate",
        "extra_charge_on_statement",
        "pending_cash_withdrawal",
        "fiat_currency_support",
        "card_delivery_estimate",
        "automatic_top_up",
        "card_not_working",
        "exchange_via_app",
        "lost_or_stolen_card",
        "age_limit",
        "pin_blocked",
        "contactless_not_working",
        "top_up_by_bank_transfer_charge",
        "pending_top_up",
        "cancel_transfer",
        "top_up_limits",
        "wrong_amount_of_cash_received",
        "card_payment_fee_charged",
        "transfer_not_received_by_recipient",
        "supported_cards_and_currencies",
        "getting_virtual_card",
        "card_acceptance",
        "top_up_reverted",
        "balance_not_updated_after_cheque_or_cash_deposit",
        "card_payment_not_recognised",
        "edit_personal_details",
        "why_verify_identity",
        "unable_to_verify_identity",
        "get_physical_card",
        "visa_or_mastercard",
        "topping_up_by_card",
        "disposable_card_limits",
        "compromised_card",
        "atm_support",
        "direct_debit_payment_not_recognised",
        "passcode_forgotten",
        "declined_cash_withdrawal",
        "pending_card_payment",
        "lost_or_stolen_phone",
        "request_refund",
        "declined_transfer",
        "Refund_not_showing_up",
        "declined_card_payment",
        "pending_transfer",
        "terminate_account",
        "card_swallowed",
        "transaction_charged_twice",
        "verify_source_of_funds",
        "transfer_timing",
        "reverted_card_payment?",
        "change_pin",
        "beneficiary_not_allowed",
        "transfer_fee_charged",
        "receiving_money",
        "failed_transfer",
        "transfer_into_account",
        "verify_top_up",
        "getting_spare_card",
        "top_up_by_cash_or_cheque",
        "order_physical_card",
        "virtual_card_not_working",
        "wrong_exchange_rate_for_cash_withdrawal",
        "get_disposable_virtual_card",
        "top_up_failed",
        "balance_not_updated_after_bank_transfer",
        "cash_withdrawal_not_recognised",
        "exchange_charge",
        "top_up_by_card_charge",
        "activate_my_card",
        "cash_withdrawal_charge",
        "card_about_to_expire",
        "apple_pay_or_google_pay",
        "verify_my_identity",
        "country_support",
    ]
)


@dataclass(frozen=True)
class SourceSpec:
    filename: str
    sha256: str
    rows: int
    categories: frozenset[str]

    @property
    def url(self) -> str:
        return f"https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/{REVISION}/banking_data/{self.filename}"


SOURCES = (
    SourceSpec(
        "train.csv",
        "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b",
        10003,
        CATEGORIES,
    ),
    SourceSpec(
        "test.csv",
        "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d",
        3080,
        CATEGORIES,
    ),
)
