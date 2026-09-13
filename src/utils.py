"""utils.py — small reusable helpers shared across notebooks."""


def get_heatmap_data(df, col1='Booking Status', col2='Day-Night'):
    """Group by two categorical columns and pivot into a 2D count table."""
    return df.groupby([col1, col2]).size().unstack()
