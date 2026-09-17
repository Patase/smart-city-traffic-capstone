import logging
import sys
import pandas as pd

from pathlib import Path

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


# ============================================================
# File paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = (
    BASE_DIR
    / "data"
    / "part3_ml_data.csv"
)

RESULTS_DIR = (
    BASE_DIR
    / "results"
)

ALL_RULES_FILE = (
    RESULTS_DIR
    / "association_rules.csv"
)

TOP_RULES_FILE = (
    RESULTS_DIR
    / "top_association_rules.csv"
)

LOG_FILE = (
    BASE_DIR
    / "part3_ml.log"
)


# ============================================================
# Logger setup
# ============================================================

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:

    console_handler = logging.StreamHandler()

    file_handler = logging.FileHandler(
        LOG_FILE,
        mode="a"
    )

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(module)s | %(message)s"
    )

    console_handler.setFormatter(
        formatter
    )

    file_handler.setFormatter(
        formatter
    )

    logger.addHandler(
        console_handler
    )

    logger.addHandler(
        file_handler
    )

logger.propagate = False


# ============================================================
# Load dataset
# ============================================================

def load_data():

    try:

        df = pd.read_csv(
            DATA_FILE
        )

        logger.info(
            "Part 3 dataset loaded successfully: %d rows, %d columns.",
            df.shape[0],
            df.shape[1]
        )

        return df

    except Exception as e:

        logger.error(
            "Unable to load dataset: %s",
            e
        )

        sys.exit(1)


# ============================================================
# Create categories for association-rule analysis
# ============================================================

def prepare_categories(df):

    df = df.copy()


    # --------------------------------------------------------
    # Time of day
    # --------------------------------------------------------

    def get_time_of_day(hour):

        if hour < 6:

            return "Night"

        elif hour < 12:

            return "Morning"

        elif hour < 18:

            return "Afternoon"

        else:

            return "Evening"


    df["time_of_day"] = (
        df["hour"]
        .apply(get_time_of_day)
    )


    # --------------------------------------------------------
    # Weekday / Weekend
    # --------------------------------------------------------

    df["weekday_type"] = (

        df["is_weekend"]

        .map({
            0: "Weekday",
            1: "Weekend"
        })

    )


    logger.info(
        "Association-rule categories prepared."
    )


    return df


# ============================================================
# Display category counts
# ============================================================

def show_category_counts(df):

    print("\n========================================")
    print("ASSOCIATION RULE DATA SUMMARY")
    print("========================================")


    print("\nTime of day:")

    print(
        df["time_of_day"]
        .value_counts()
        .to_string()
    )


    print("\nWeekday type:")

    print(
        df["weekday_type"]
        .value_counts()
        .to_string()
    )


    print("\nCongestion category:")

    print(
        df["congestion_category_ml"]
        .value_counts()
        .to_string()
    )


# ============================================================
# Build transactions
# ============================================================

def create_transactions(df):

    transactions = []


    for _, row in df.iterrows():

        transaction = [

            "time=" + str(
                row["time_of_day"]
            ),

            "day=" + str(
                row["weekday_type"]
            ),

            "weather=" + str(
                row["weather_main"]
            ),

            "congestion=" + str(
                row["congestion_category_ml"]
            )

        ]


        transactions.append(
            transaction
        )


    logger.info(
        "%d transactions created.",
        len(transactions)
    )


    return transactions


# ============================================================
# Convert transactions to one-hot format
# ============================================================

def encode_transactions(transactions):

    encoder = TransactionEncoder()


    encoded_array = (
        encoder
        .fit(transactions)
        .transform(transactions)
    )


    encoded_df = pd.DataFrame(

        encoded_array,

        columns=encoder.columns_

    )


    logger.info(
        "Transactions encoded into %d binary items.",
        encoded_df.shape[1]
    )


    return encoded_df


# ============================================================
# Find frequent itemsets
# ============================================================

def find_frequent_itemsets(encoded_df):

    frequent_itemsets = apriori(

        encoded_df,

        min_support=0.01,

        use_colnames=True

    )


    logger.info(
        "%d frequent itemsets identified.",
        len(frequent_itemsets)
    )


    return frequent_itemsets


# ============================================================
# Generate association rules
# ============================================================

def generate_rules(frequent_itemsets):

    if len(frequent_itemsets) == 0:

        logger.error(
            "No frequent itemsets were found."
        )

        sys.exit(1)


    rules = association_rules(

        frequent_itemsets,

        metric="lift",

        min_threshold=1.0

    )


    logger.info(
        "%d total association rules generated.",
        len(rules)
    )


    return rules


# ============================================================
# Keep rules predicting congestion
# ============================================================

def filter_congestion_rules(rules):

    congestion_rules = []


    for _, row in rules.iterrows():

        antecedents = row[
            "antecedents"
        ]

        consequents = row[
            "consequents"
        ]


        # Consequent must contain exactly one item
        if len(consequents) != 1:

            continue


        consequent_item = next(
            iter(consequents)
        )


        # We only want rules predicting congestion
        if not consequent_item.startswith(
            "congestion="
        ):

            continue


        # Do not allow congestion on the left-hand side
        contains_congestion = any(

            item.startswith(
                "congestion="
            )

            for item in antecedents

        )


        if contains_congestion:

            continue


        congestion_rules.append(
            row
        )


    if len(congestion_rules) == 0:

        logger.warning(
            "No congestion-predicting rules were found."
        )

        return pd.DataFrame(
            columns=rules.columns
        )


    congestion_rules_df = pd.DataFrame(
        congestion_rules
    )


    congestion_rules_df = (

        congestion_rules_df

        .sort_values(
            by="lift",
            ascending=False
        )

        .reset_index(drop=True)

    )


    logger.info(
        "%d congestion-predicting rules retained.",
        len(congestion_rules_df)
    )


    return congestion_rules_df


# ============================================================
# Make rules easier to read
# ============================================================

def make_readable_rules(rules_df):

    readable_df = (
        rules_df.copy()
    )


    readable_df[
        "antecedents"
    ] = readable_df[
        "antecedents"
    ].apply(

        lambda items:
        ", ".join(
            sorted(list(items))
        )

    )


    readable_df[
        "consequents"
    ] = readable_df[
        "consequents"
    ].apply(

        lambda items:
        ", ".join(
            sorted(list(items))
        )

    )


    return readable_df


# ============================================================
# Save results
# ============================================================

def save_results(rules_df):

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    readable_rules = (
        make_readable_rules(
            rules_df
        )
    )


    # Save all congestion-predicting rules
    readable_rules.to_csv(

        ALL_RULES_FILE,

        index=False

    )


    # Save top 10 by lift
    top_rules = (

        readable_rules

        .head(10)

        .copy()

    )


    top_rules.to_csv(

        TOP_RULES_FILE,

        index=False

    )


    logger.info(
        "Association rules saved: %s",
        ALL_RULES_FILE
    )


    logger.info(
        "Top association rules saved: %s",
        TOP_RULES_FILE
    )


    return top_rules


# ============================================================
# Display top rules
# ============================================================

def display_top_rules(top_rules):

    print("\n========================================")
    print("TOP ASSOCIATION RULES")
    print("========================================")


    if len(top_rules) == 0:

        print(
            "\nNo congestion-predicting rules found."
        )

        return


    display_columns = [

        "antecedents",
        "consequents",
        "support",
        "confidence",
        "lift"

    ]


    display_df = (

        top_rules[
            display_columns
        ]

        .copy()

    )


    display_df[
        "support"
    ] = display_df[
        "support"
    ].round(4)


    display_df[
        "confidence"
    ] = display_df[
        "confidence"
    ].round(4)


    display_df[
        "lift"
    ] = display_df[
        "lift"
    ].round(4)


    print(
        "\n"
        + display_df.to_string(
            index=False
        )
    )


# ============================================================
# Main program
# ============================================================

def main():

    logger.info(
        "Association rule mining started."
    )


    # Load dataset
    # ========================================================

    df = load_data()


    # Create categories
    # ========================================================

    df = prepare_categories(
        df
    )


    # Show basic category counts
    # ========================================================

    show_category_counts(
        df
    )


    # Create transactions
    # ========================================================

    transactions = (
        create_transactions(
            df
        )
    )


    print(
        f"\nNumber of transactions: "
        f"{len(transactions)}"
    )


    # One-hot encode
    # ========================================================

    encoded_df = (
        encode_transactions(
            transactions
        )
    )


    # Frequent itemsets
    # ========================================================

    frequent_itemsets = (
        find_frequent_itemsets(
            encoded_df
        )
    )


    print(
        f"Frequent itemsets: "
        f"{len(frequent_itemsets)}"
    )


    # Association rules
    # ========================================================

    rules = (
        generate_rules(
            frequent_itemsets
        )
    )


    print(
        f"Total rules generated: "
        f"{len(rules)}"
    )


    # Keep congestion prediction rules
    # ========================================================

    congestion_rules = (
        filter_congestion_rules(
            rules
        )
    )


    print(
        f"Congestion-predicting rules: "
        f"{len(congestion_rules)}"
    )


    # Save and display top rules
    # ========================================================

    top_rules = (
        save_results(
            congestion_rules
        )
    )


    display_top_rules(
        top_rules
    )


    logger.info(
        "Association rule mining completed successfully."
    )


if __name__ == "__main__":

    main()