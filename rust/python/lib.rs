use pyo3::prelude::*;

mod example;

pub use example::Example;

fn map_err(error: String) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(error)
}

#[pyfunction]
fn sample_covariance_matrix(data: Vec<Vec<f64>>, frequency: f64) -> PyResult<Vec<Vec<f64>>> {
    ::finance_risk::sample_covariance_matrix(data, frequency).map_err(map_err)
}

#[pyfunction]
fn ledoit_wolf_covariance_matrix(
    data: Vec<Vec<f64>>,
    frequency: f64,
) -> PyResult<(Vec<Vec<f64>>, f64)> {
    ::finance_risk::ledoit_wolf_covariance_matrix(data, frequency).map_err(map_err)
}

#[pyfunction]
fn oas_covariance_matrix(data: Vec<Vec<f64>>, frequency: f64) -> PyResult<(Vec<Vec<f64>>, f64)> {
    ::finance_risk::oas_covariance_matrix(data, frequency).map_err(map_err)
}

#[pyfunction]
fn covariance_to_correlation(covariance: Vec<Vec<f64>>) -> PyResult<Vec<Vec<f64>>> {
    ::finance_risk::covariance_to_correlation(covariance).map_err(map_err)
}

#[pyfunction]
fn correlation_to_covariance(
    correlation: Vec<Vec<f64>>,
    volatility: Vec<f64>,
) -> PyResult<Vec<Vec<f64>>> {
    ::finance_risk::correlation_to_covariance(correlation, volatility).map_err(map_err)
}

#[pyfunction]
fn rolling_correlation(x: Vec<f64>, y: Vec<f64>, window: usize) -> PyResult<Vec<f64>> {
    ::finance_risk::rolling_correlation(x, y, window).map_err(map_err)
}

#[pyfunction]
fn cointegration_spread(y: Vec<f64>, x: Vec<f64>) -> PyResult<(f64, f64, Vec<f64>)> {
    ::finance_risk::cointegration_spread(y, x).map_err(map_err)
}

#[pyfunction]
fn transfer_entropy(source: Vec<f64>, target: Vec<f64>, bins: usize, lag: usize) -> PyResult<f64> {
    ::finance_risk::transfer_entropy(source, target, bins, lag).map_err(map_err)
}

#[pyfunction]
fn twap_schedule(total_quantity: f64, periods: usize) -> PyResult<Vec<f64>> {
    ::finance_risk::twap_schedule(total_quantity, periods).map_err(map_err)
}

#[pyfunction]
fn vwap_schedule(total_quantity: f64, volume_curve: Vec<f64>) -> PyResult<Vec<f64>> {
    ::finance_risk::vwap_schedule(total_quantity, volume_curve).map_err(map_err)
}

#[pyfunction]
fn almgren_chriss_market_impact(
    quantity: f64,
    volatility: f64,
    daily_volume: f64,
    spread: f64,
    temporary_impact: f64,
    permanent_impact: f64,
) -> PyResult<(f64, f64, f64, f64)> {
    ::finance_risk::almgren_chriss_market_impact(
        quantity,
        volatility,
        daily_volume,
        spread,
        temporary_impact,
        permanent_impact,
    )
    .map_err(map_err)
}

#[pyfunction]
fn optimal_execution_trajectory(
    total_quantity: f64,
    periods: usize,
    risk_aversion: f64,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    ::finance_risk::optimal_execution_trajectory(total_quantity, periods, risk_aversion)
        .map_err(map_err)
}

#[pyfunction]
fn factor_risk_decomposition(
    weights: Vec<f64>,
    factor_loadings: Vec<Vec<f64>>,
    factor_covariance: Vec<Vec<f64>>,
    specific_variance: Vec<f64>,
) -> PyResult<(f64, f64, f64, Vec<f64>)> {
    ::finance_risk::factor_risk_decomposition(
        weights,
        factor_loadings,
        factor_covariance,
        specific_variance,
    )
    .map_err(map_err)
}

#[pymodule]
fn finance_risk(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    // Example
    m.add_class::<Example>().unwrap();
    m.add_function(wrap_pyfunction!(sample_covariance_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(ledoit_wolf_covariance_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(oas_covariance_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(covariance_to_correlation, m)?)?;
    m.add_function(wrap_pyfunction!(correlation_to_covariance, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_correlation, m)?)?;
    m.add_function(wrap_pyfunction!(cointegration_spread, m)?)?;
    m.add_function(wrap_pyfunction!(transfer_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(twap_schedule, m)?)?;
    m.add_function(wrap_pyfunction!(vwap_schedule, m)?)?;
    m.add_function(wrap_pyfunction!(almgren_chriss_market_impact, m)?)?;
    m.add_function(wrap_pyfunction!(optimal_execution_trajectory, m)?)?;
    m.add_function(wrap_pyfunction!(factor_risk_decomposition, m)?)?;
    Ok(())
}
