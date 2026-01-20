"""
GoldenHour Utility Package

Utility functions for time series forecasting and safe-haven hypothesis testing.
"""

from .data_utils import (
    load_gold_silver_data,
    validate_dataset,
    test_stationarity,
    print_stationarity_results,
    calculate_technical_indicators,
    create_sequences_for_lstm,
    calculate_naive_baseline,
    check_for_zero_or_negative_prices,
    split_train_test_temporal,
    create_advanced_features
)

from .forecast_utils import (
    convert_log_returns_to_prices,
    evaluate_forecast_prices,
    check_exog_alignment,
    create_lagged_exog,
    walk_forward_validation,
    calculate_diebold_mariano,
    create_forecast_comparison_table,
    print_forecast_summary,
    calculate_risk_adjusted_metrics,
    print_risk_adjusted_summary
)

from .model_selection import (
    grid_search_arima,
    compare_models_ic,
    validate_residuals,
    print_diagnostics_report,
    rolling_window_validation,
    granger_causality_test
)

from .deep_learning_utils import (
    TimeSeriesEarlyStopping,
    LearningRateScheduler,
    find_optimal_batch_size,
    plot_training_history,
    calculate_directional_accuracy,
    lstm_architecture_search,
    create_ensemble_predictions,
    analyze_lstm_feature_importance
)

__version__ = '1.2.0'

__all__ = [
    # Data utilities
    'load_gold_silver_data',
    'validate_dataset',
    'test_stationarity',
    'print_stationarity_results',
    'calculate_technical_indicators',
    'create_sequences_for_lstm',
    'calculate_naive_baseline',
    'check_for_zero_or_negative_prices',
    'split_train_test_temporal',
    'create_advanced_features',
    
    # Forecast utilities
    'convert_log_returns_to_prices',
    'evaluate_forecast_prices',
    'check_exog_alignment',
    'create_lagged_exog',
    'walk_forward_validation',
    'calculate_diebold_mariano',
    'create_forecast_comparison_table',
    'print_forecast_summary',
    'calculate_risk_adjusted_metrics',
    'print_risk_adjusted_summary',
    
    # Model selection utilities
    'grid_search_arima',
    'compare_models_ic',
    'validate_residuals',
    'print_diagnostics_report',
    'rolling_window_validation',
    'granger_causality_test',
    
    # Deep learning utilities
    'TimeSeriesEarlyStopping',
    'LearningRateScheduler',
    'find_optimal_batch_size',
    'plot_training_history',
    'calculate_directional_accuracy',
    'lstm_architecture_search',
    'create_ensemble_predictions',
    'analyze_lstm_feature_importance'
]
