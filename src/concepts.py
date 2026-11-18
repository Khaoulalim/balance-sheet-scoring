# src/features/balance_sheet_concepts.py

BALANCE_SHEET_CONCEPTS = [
    # Core totals
    "Assets",
    "AssetsCurrent",
    "AssetsNoncurrent",
    "Liabilities",
    "LiabilitiesCurrent",
    "LiabilitiesNoncurrent",
    "StockholdersEquity",
    "LiabilitiesAndStockholdersEquity",

    # Cash & working capital
    "CashAndCashEquivalentsAtCarryingValue",
    "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    "AccountsReceivableNetCurrent",
    "InventoryNet",
    "PrepaidExpenseCurrent",

    # PP&E and intangibles
    "PropertyPlantAndEquipmentNet",
    "Goodwill",
    "IntangibleAssetsNetExcludingGoodwill",

    # Debt
    "DebtCurrent",
    "LongTermDebt",
    "LongTermDebtCurrent",

    # Payables / accruals
    "AccountsPayableCurrent",
    "AccruedLiabilitiesCurrent",

    # Equity components (optional but useful)
    "RetainedEarningsAccumulatedDeficit",
    "CommonStockValue",
    "AdditionalPaidInCapital",
    "AccumulatedOtherComprehensiveIncomeLossNetOfTax",
]

# Postes de FLUX (compte de résultat + tableau des flux de trésorerie) :
# ils couvrent une période (start -> end), pas une date. On garde les périodes d'environ 1 an.
FLOW_CONCEPTS = [
    # Compte de résultat
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",   # chiffre d'affaires (norme ASC 606, depuis 2018)
    "SalesRevenueNet",                                       # chiffre d'affaires (ancienne étiquette)
    "OperatingIncomeLoss",                                   # résultat d'exploitation
    "NetIncomeLoss",                                         # résultat net
    "InterestExpense",                                       # charges d'intérêts
    "DepreciationDepletionAndAmortization",                  # amortissements

    # Flux de trésorerie
    "NetCashProvidedByUsedInOperatingActivities",            # cash généré par l'activité
    "NetCashProvidedByUsedInInvestingActivities",            # cash investi
    "NetCashProvidedByUsedInFinancingActivities",            # cash de financement
    "PaymentsToAcquirePropertyPlantAndEquipment",            # investissements (capex)
    "PaymentsOfDividends",                                   # dividendes versés
    "PaymentsForRepurchaseOfCommonStock",                    # rachats d'actions
]
