from datasets import load_dataset, ClassLabel, DatasetDict

# If no validation split exists, create one from the training split
DEFAULT_VAL_SPLIT_RATIO = 0.1
DEFAULT_SPLIT_SEED = 42

# Central Registry for all Datasets in the Study
DATASET_REGISTRY = {
    'emotion': {
        'hf_path': 'dair-ai/emotion',
        'hf_config': 'split',
        'text_col': 'text',
        'label_col': 'label',
        'num_labels': 6,
        'label_names': {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}
    },
    'ag_news': {
        'hf_path': 'fancyzhx/ag_news',
        'hf_config': None,
        'text_col': 'text',
        'label_col': 'label',
        'num_labels': 4,
        'label_names': {0: 'World', 1: 'Sports', 2: 'Business', 3: 'Sci/Tech'}
    },
    'banking77':{
        'hf_path': 'mteb/banking77',
        'hf_config': None,
        'text_col': 'text',
        'label_col': 'label',
        'num_labels': 77,
        'label_names': {
            0: 'activate_my_card', 1: 'age_limit', 2: 'apple_pay_or_google_pay', 3: 'atm_support',
            4: 'automatic_top_up', 5: 'balance_not_updated_after_bank_transfer',
            6: 'balance_not_updated_after_cheque_or_cash_deposit', 7: 'beneficiary_not_allowed',
            8: 'cancel_transfer', 9: 'card_about_to_expire', 10: 'card_acceptance', 11: 'card_arrival',
            12: 'card_delivery_estimate', 13: 'card_linking', 14: 'card_not_working',
            15: 'card_payment_fee_charged', 16: 'card_payment_not_recognised',
            17: 'card_payment_wrong_exchange_rate', 18: 'card_swallowed', 19: 'cash_withdrawal_charge',
            20: 'cash_withdrawal_not_recognised', 21: 'change_pin', 22: 'compromised_card',
            23: 'contactless_not_working', 24: 'country_support', 25: 'declined_card_payment',
            26: 'declined_cash_withdrawal', 27: 'declined_transfer',
            28: 'direct_debit_payment_not_recognised', 29: 'disposable_card_limits',
            30: 'edit_personal_details', 31: 'exchange_charge', 32: 'exchange_rate',
            33: 'exchange_via_app', 34: 'extra_charge_on_statement', 35: 'failed_transfer',
            36: 'fiat_currency_support', 37: 'get_disposable_virtual_card', 38: 'get_physical_card',
            39: 'getting_spare_card', 40: 'getting_virtual_card', 41: 'lost_or_stolen_card',
            42: 'lost_or_stolen_phone', 43: 'order_physical_card', 44: 'passcode_forgotten',
            45: 'pending_card_payment', 46: 'pending_cash_withdrawal', 47: 'pending_top_up',
            48: 'pending_transfer', 49: 'pin_blocked', 50: 'receiving_money',
            51: 'Refund_not_showing_up', 52: 'request_refund', 53: 'reverted_card_payment?',
            54: 'supported_cards_and_currencies', 55: 'terminate_account',
            56: 'top_up_by_bank_transfer_charge', 57: 'top_up_by_card_charge',
            58: 'top_up_by_cash_or_cheque', 59: 'top_up_failed', 60: 'top_up_limits',
            61: 'top_up_reverted', 62: 'topping_up_by_card', 63: 'transaction_charged_twice',
            64: 'transfer_fee_charged', 65: 'transfer_into_account',
            66: 'transfer_not_received_by_recipient', 67: 'transfer_timing',
            68: 'unable_to_verify_identity', 69: 'verify_my_identity',
            70: 'verify_source_of_funds', 71: 'verify_top_up', 72: 'virtual_card_not_working',
            73: 'visa_or_mastercard', 74: 'why_verify_identity', 75: 'wrong_amount_of_cash_received',
            76: 'wrong_exchange_rate_for_cash_withdrawal',
        }
    },
    '20_newsgroups': {
        'hf_path': 'SetFit/20_newsgroups',
        'hf_config': None,
        'text_col': 'text',
        'label_col': 'label',
        'num_labels': 20,
        'label_names': {
            0: 'alt.atheism', 1: 'comp.graphics', 2: 'comp.os.ms-windows.misc',
            3: 'comp.sys.ibm.pc.hardware', 4: 'comp.sys.mac.hardware', 5: 'comp.windows.x',
            6: 'misc.forsale', 7: 'rec.autos', 8: 'rec.motorcycles', 9: 'rec.sport.baseball',
            10: 'rec.sport.hockey', 11: 'sci.crypt', 12: 'sci.electronics', 13: 'sci.med',
            14: 'sci.space', 15: 'soc.religion.christian', 16: 'talk.politics.guns',
            17: 'talk.politics.mideast', 18: 'talk.politics.misc', 19: 'talk.religion.misc'
        }
    },
    # Future datasets can be added here seamlessly
}


def get_dataset_info(dataset_name: str):
    """Retrieve metadata for a dataset from the registry."""
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Dataset '{dataset_name}' not registered. Available: {list(DATASET_REGISTRY.keys())}")
    return DATASET_REGISTRY[dataset_name]


def drop_missing_rows(raw_ds: DatasetDict, columns: list) -> DatasetDict:
    """Removes rows with a null value in any of the given columns, per split."""
    return DatasetDict({
        split: split_ds.filter(lambda ex: all(ex[c] is not None for c in columns))
        for split, split_ds in raw_ds.items()
    })


def normalize_whitespace(raw_ds: DatasetDict, text_col: str) -> DatasetDict:
    """Strips leading/trailing whitespace and collapses internal whitespace runs in the text column, then drops rows left empty."""
    ds = DatasetDict({
        split: split_ds.map(lambda ex: {text_col: ' '.join(str(ex[text_col]).split())})
        for split, split_ds in raw_ds.items()
    })
    return DatasetDict({
        split: split_ds.filter(lambda ex: ex[text_col] != '')
        for split, split_ds in ds.items()
    })


def drop_duplicate_rows(raw_ds: DatasetDict, text_col: str) -> DatasetDict:
    """Removes exact duplicate rows within each split, keeping the first occurrence (by text)."""
    deduped = {}
    for split, split_ds in raw_ds.items():
        seen = set()

        def _is_first_occurrence(ex):
            if ex[text_col] in seen:
                return False
            seen.add(ex[text_col])
            return True

        deduped[split] = split_ds.filter(_is_first_occurrence)
    return DatasetDict(deduped)


def clean_dataset(raw_ds: DatasetDict, text_col: str, label_col: str) -> DatasetDict:
    """
    Basic cleaning applied uniformly across every split: drops rows with a
    missing text/label, strips/collapses whitespace in the text column, and
    removes exact duplicate rows.
    """
    ds = drop_missing_rows(raw_ds, [text_col, label_col])
    ds = normalize_whitespace(ds, text_col)
    ds = drop_duplicate_rows(ds, text_col)
    return ds


def ensure_validation_split(
    raw_ds: DatasetDict,
    label_col: str,
    num_labels: int,
    val_ratio: float = DEFAULT_VAL_SPLIT_RATIO,
    seed: int = DEFAULT_SPLIT_SEED,
) -> DatasetDict:
    """
    Guarantees a 'validation' split exists. If the dataset already has one,
    it's returned unchanged. Otherwise a stratified `val_ratio` slice is
    carved out of the training split only — the test split is never touched.
    """
    if 'validation' in raw_ds:
        return raw_ds

    train_ds = raw_ds['train']

    # train_test_split can only stratify on a ClassLabel column. Cast using
    # an explicit numeric-ordered names list so label ints are preserved
    # exactly (datasets' class_encode_column instead sorts names as strings,
    # which silently remaps e.g. label 63 -> 60 once "10".."19" sort before "2").
    original_label_feature = train_ds.features[label_col]
    if isinstance(original_label_feature, ClassLabel):
        strat_ds = train_ds
    else:
        strat_ds = train_ds.cast_column(label_col, ClassLabel(names=[str(i) for i in range(num_labels)]))

    split = strat_ds.train_test_split(test_size=val_ratio, stratify_by_column=label_col, seed=seed)
    new_train, new_val = split['train'], split['test']

    # Cast back to the original label dtype so train/validation match test.
    if not isinstance(original_label_feature, ClassLabel):
        new_train = new_train.cast_column(label_col, original_label_feature)
        new_val = new_val.cast_column(label_col, original_label_feature)

    new_ds = {'train': new_train, 'validation': new_val}
    for split_name, split_ds in raw_ds.items():
        if split_name != 'train':
            new_ds[split_name] = split_ds

    return DatasetDict(new_ds)


def load_and_clean_raw_dataset(
    dataset_name: str,
    val_split_ratio: float = DEFAULT_VAL_SPLIT_RATIO,
    seed: int = DEFAULT_SPLIT_SEED,
):
    """
    Loads a registered dataset from Hugging Face, applies cleaning, and
    guarantees a validation split. Returns the raw (untokenized) DatasetDict
    alongside its registry info, so callers (training prep, EDA) share one
    canonical view of the data.
    """
    info = get_dataset_info(dataset_name)

    if info['hf_config']:
        raw_ds = load_dataset(info['hf_path'], info['hf_config'])
    else:
        raw_ds = load_dataset(info['hf_path'])

    raw_ds = clean_dataset(raw_ds, info['text_col'], info['label_col'])
    raw_ds = ensure_validation_split(raw_ds, info['label_col'], info['num_labels'], val_split_ratio, seed)

    return raw_ds, info


def load_and_prep_dataset(
    dataset_name: str,
    tokenizer,
    max_seq_len: int = 64,
    val_split_ratio: float = DEFAULT_VAL_SPLIT_RATIO,
    seed: int = DEFAULT_SPLIT_SEED,
):
    """
    Loads dataset from Hugging Face, cleans it, guarantees a validation split,
    applies tokenization with max_length padding, renames label column, sets
    PyTorch format, and returns tokenized dataset splits.
    """
    raw_ds, info = load_and_clean_raw_dataset(dataset_name, val_split_ratio, seed)

    text_col = info['text_col']
    label_col = info['label_col']

    def tokenize_function(batch):
        return tokenizer(
            batch[text_col],
            truncation=True,
            padding='max_length',
            max_length=max_seq_len,
        )

    # Tokenize and remove raw text column to prevent collator issues
    tokenized_ds = raw_ds.map(
        tokenize_function,
        batched=True,
        remove_columns=[text_col]
    )

    # Ensure label column is named 'labels' for HuggingFace Trainer
    if label_col != 'labels':
        tokenized_ds = tokenized_ds.rename_column(label_col, 'labels')

    # Set PyTorch tensor format
    tokenized_ds.set_format('torch')

    return tokenized_ds, info['num_labels'], info['label_names']
